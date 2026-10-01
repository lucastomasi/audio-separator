"""PWA manifest, icons, and the tags Safari reads before any page script runs."""
import os
import socket
import unittest
import urllib.request

from PIL import Image
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient

import pwa_web

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PWA_DIR = os.path.join(ROOT, "pwa")
ICON_SIZES = {
    "icon-180.png": 180,
    "icon-192.png": 192,
    "icon-512.png": 512,
    "icon-maskable-512.png": 512,
    "favicon-32.png": 32,
}


class IconTests(unittest.TestCase):
    def test_committed_icons_match_the_home_screen_sizes(self):
        for name, size in ICON_SIZES.items():
            path = os.path.join(PWA_DIR, name)
            with Image.open(path) as image:
                self.assertEqual(image.size, (size, size), name)
                self.assertEqual(image.format, "PNG", name)
        ico = os.path.join(PWA_DIR, "favicon.ico")
        with Image.open(ico) as image:
            self.assertEqual(image.format, "ICO")


class ManifestTests(unittest.TestCase):
    def test_manifest_is_an_installable_web_app(self):
        data = pwa_web.manifest()
        self.assertEqual(data["name"], "Audio Separator")
        self.assertEqual(data["short_name"], "Separator")
        self.assertEqual(data["display"], "standalone")
        self.assertEqual(data["start_url"], "/")
        self.assertEqual(data["scope"], "/")
        self.assertEqual(data["theme_color"], "#4f46e5")
        self.assertEqual(data["background_color"], "#eef2ff")
        self.assertFalse(data["prefer_related_applications"])
        self.assertNotIn("related_applications", data)
        sizes = {icon["sizes"] for icon in data["icons"]}
        self.assertIn("180x180", sizes)
        self.assertIn("192x192", sizes)
        self.assertIn("512x512", sizes)


class HeadInjectionTests(unittest.TestCase):
    def test_apple_tags_land_in_the_original_head(self):
        html = b"<html><head><title>Audio Separator</title></head><body></body></html>"
        updated = pwa_web.inject_pwa_head(html)
        head, _, _rest = updated.partition(b"</head>")
        self.assertIn(b"apple-mobile-web-app-capable", head)
        self.assertIn(b"mobile-web-app-capable", head)
        self.assertIn(b'content="Separator"', head)
        self.assertIn(b'rel="apple-touch-icon"', head)
        self.assertIn(b'content="#4f46e5"', head)
        self.assertIn(b"/sw.js", head)
        self.assertEqual(pwa_web.inject_pwa_head(updated), updated)

    def test_service_worker_does_not_capture_audio_requests(self):
        script = pwa_web.service_worker_js()
        self.assertIn('addEventListener("fetch"', script)
        self.assertNotIn("respondWith", script)
        self.assertNotIn("caches", script)


class InstalledAppTests(unittest.TestCase):
    def test_routes_and_html_injection(self):
        app = FastAPI()

        @app.get("/")
        def home():
            return HTMLResponse("<html><head><title>t</title></head><body>ok</body></html>")

        pwa_web.install_pwa(app)
        client = TestClient(app)

        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("apple-mobile-web-app-capable", page.text)
        self.assertLess(page.text.index("apple-touch-icon"), page.text.index("</head>"))

        manifest = client.get("/manifest.webmanifest")
        self.assertEqual(manifest.status_code, 200)
        self.assertIn("application/manifest+json", manifest.headers["content-type"])
        self.assertEqual(manifest.json()["display"], "standalone")
        self.assertEqual(client.get("/manifest.json").json()["short_name"], "Separator")

        icon = client.get("/apple-touch-icon.png")
        self.assertEqual(icon.status_code, 200)
        self.assertTrue(icon.content.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(client.get("/pwa/icon-512.png").status_code, 200)
        self.assertIn("serviceWorker.register", page.text)
        worker = client.get("/sw.js")
        self.assertEqual(worker.status_code, 200)
        self.assertIn('addEventListener("fetch"', worker.text)
        self.assertNotIn("respondWith", worker.text)


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class GradioLaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import launch_server

        cls.port = _free_port()
        cls.launch = launch_server(
            server_name="127.0.0.1",
            server_port=cls.port,
            prevent_thread_lock=True,
            quiet=True,
        )
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        app = cls.launch[0]
        app.get_blocks().close()

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as response:
            return response.status, response.headers, response.read()

    def test_gradio_page_has_apple_tags_before_scripts_run(self):
        status, _headers, body = self._get("/")
        self.assertEqual(status, 200)
        head = body.split(b"</head>", 1)[0]
        self.assertIn(b"apple-mobile-web-app-capable", head)
        self.assertIn(b'rel="apple-touch-icon"', head)
        self.assertIn(b'rel="manifest"', head)
        self.assertIn(b"/sw.js", head)

        status, headers, manifest_body = self._get("/manifest.json")
        self.assertEqual(status, 200)
        self.assertIn("application/manifest+json", headers["content-type"])
        self.assertIn(b'"display":"standalone"', manifest_body.replace(b" ", b""))
        self.assertIn(b'"short_name":"Separator"', manifest_body.replace(b" ", b""))
        self.assertNotIn(b"App Store", manifest_body)
        self.assertNotIn(b".ipa", manifest_body)

        status, icon_headers, icon = self._get("/pwa/icon-192.png")
        self.assertEqual(status, 200)
        self.assertTrue(icon.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn("image/png", icon_headers["content-type"])
