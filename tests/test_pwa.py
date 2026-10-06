"""PWA tags in the HTML Safari reads, and static files for Add to Home Screen."""
import json
import os
import socket
import unittest
import urllib.error
import urllib.request

import app
from pwa_server import inject_pwa_html


def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _get(url, token=None):
    headers = {}
    if token:
        headers[app.TOKEN_HEADER] = token
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()


class InjectTests(unittest.TestCase):
    def test_apple_tags_land_in_the_initial_html(self):
        html = (
            "<html><head>"
            '<link rel="manifest" href="/manifest.json" />'
            "</head><body></body></html>"
        )
        out = inject_pwa_html(html)
        self.assertIn('content="yes"', out)
        self.assertIn("apple-mobile-web-app-capable", out)
        self.assertIn("apple-touch-icon", out)
        self.assertIn('href="/pwa/manifest.webmanifest"', out)
        self.assertNotIn('href="/manifest.json"', out)
        self.assertEqual(inject_pwa_html(out).count("apple-mobile-web-app-capable"), 1)


class ServerTests(unittest.TestCase):
    def setUp(self):
        self._old_token = os.environ.get(app.TOKEN_ENV)
        os.environ[app.TOKEN_ENV] = "c" * 32
        self.token = os.environ[app.TOKEN_ENV]
        self.port = _free_port()
        self.demo = app.build_server()
        from pwa_server import launch_with_pwa

        launch_with_pwa(
            self.demo,
            app.launch_kwargs(
                prevent_thread_lock=True,
                quiet=True,
                server_name="127.0.0.1",
                server_port=self.port,
            ),
        )

    def tearDown(self):
        self.demo.close()
        if self._old_token is None:
            os.environ.pop(app.TOKEN_ENV, None)
        else:
            os.environ[app.TOKEN_ENV] = self._old_token

    def test_phone_assets_and_locked_page(self):
        base = f"http://127.0.0.1:{self.port}"
        status, content_type, body = _get(base + "/")
        self.assertEqual(status, 401)
        self.assertNotIn(b"apple-mobile-web-app-capable", body)

        status, content_type, body = _get(base + "/", token=self.token)
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn(b"apple-mobile-web-app-capable", body)
        self.assertIn(b"apple-mobile-web-app-title", body)
        self.assertIn(b"/pwa/manifest.webmanifest", body)
        self.assertIn(b"/pwa/icon-180.png", body)
        self.assertNotIn(b'href="/manifest.json"', body)

        status, content_type, body = _get(base + "/pwa/manifest.webmanifest")
        self.assertEqual(status, 200)
        self.assertIn("manifest", content_type)
        manifest = json.loads(body)
        self.assertEqual(manifest["display"], "standalone")
        self.assertEqual(manifest["name"], "Audio Separator")
        self.assertEqual(manifest["start_url"], "/")
        self.assertNotIn(self.token.encode(), body)
        self.assertTrue(manifest["icons"])

        status, _content_type, body = _get(
            base + "/pwa/manifest.webmanifest", token=self.token
        )
        signed = json.loads(body)
        self.assertEqual(signed["start_url"], f"/?{app.TOKEN_QUERY}={self.token}")

        status, content_type, body = _get(base + "/pwa/icon-180.png")
        self.assertEqual(status, 200)
        self.assertIn("image/png", content_type)
        self.assertTrue(body.startswith(b"\x89PNG"))

        status, content_type, body = _get(base + "/sw.js")
        self.assertEqual(status, 200)
        self.assertIn(b"fetch", body)

        status, _content_type, body = _get(base + "/apple-touch-icon.png")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"\x89PNG"))
