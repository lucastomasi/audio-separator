"""Home-screen install tags for iPhone Safari. No audio models and no App Store build."""

import asyncio
import socket
import time
import unittest
from pathlib import Path

import httpx
from PIL import Image
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route

from app import launch_kwargs, phone_hint
from pwa import (
    APP_NAME,
    APP_SHORT_NAME,
    BACKGROUND_COLOR,
    MANIFEST,
    MARKER,
    PWA_DIR,
    PWAMiddleware,
    THEME_COLOR,
    inject_html,
)


GRADIO_VIEWPORT = """<meta
\t\t\tname="viewport"
\t\t\tcontent="width=device-width, initial-scale=1, shrink-to-fit=no"
\t\t/>"""


class InjectTests(unittest.TestCase):
    def test_apple_tags_land_in_the_first_html_response(self):
        html = f"<html><head><title>Audio Separator</title>{GRADIO_VIEWPORT}</head><body>hola</body></html>"
        tagged = inject_html(html)
        self.assertIn('name="apple-mobile-web-app-capable" content="yes"', tagged)
        self.assertIn('name="mobile-web-app-capable" content="yes"', tagged)
        self.assertIn(f'name="apple-mobile-web-app-title" content="{APP_SHORT_NAME}"', tagged)
        self.assertIn(f'name="theme-color" content="{THEME_COLOR}"', tagged)
        self.assertIn('rel="apple-touch-icon"', tagged)
        self.assertIn("viewport-fit=cover", tagged)
        self.assertIn("hola", tagged)
        self.assertEqual(inject_html(tagged), tagged)

    def test_leaves_non_html_alone(self):
        self.assertEqual(inject_html("sin cabeza"), "sin cabeza")


class MiddlewareTests(unittest.TestCase):
    def test_manifest_icons_and_html(self):
        async def homepage(request):
            return HTMLResponse(
                "<html><head><title>x</title></head><body>ok</body></html>"
            )

        app = PWAMiddleware(Starlette(routes=[Route("/", homepage)]))

        async def check():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                page = await client.get("/")
                self.assertEqual(page.status_code, 200)
                self.assertIn(MARKER, page.text)
                self.assertIn("apple-touch-icon.png", page.text)

                manifest = await client.get("/manifest.json")
                self.assertEqual(manifest.status_code, 200)
                self.assertIn(
                    "application/manifest+json", manifest.headers["content-type"]
                )
                body = manifest.json()
                self.assertEqual(body["name"], APP_NAME)
                self.assertEqual(body["short_name"], APP_SHORT_NAME)
                self.assertEqual(body["display"], "standalone")
                self.assertEqual(body["theme_color"], THEME_COLOR)
                self.assertEqual(body["background_color"], BACKGROUND_COLOR)
                self.assertFalse(body["prefer_related_applications"])
                self.assertGreaterEqual(len(body["icons"]), 2)
                alias = await client.get("/manifest.webmanifest")
                self.assertEqual(alias.json()["name"], APP_NAME)

                icon = await client.get("/apple-touch-icon.png")
                self.assertEqual(icon.status_code, 200)
                self.assertTrue(icon.headers["content-type"].startswith("image/png"))
                self.assertEqual(icon.content[:8], b"\x89PNG\r\n\x1a\n")
                script = await client.get("/sw.js")
                self.assertEqual(script.status_code, 200)
                self.assertIn("serviceWorker", page.text)
                self.assertIn("navigate", script.text)

        asyncio.run(check())

    def test_event_stream_is_not_buffered(self):
        events = []

        async def downstream(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            await send(
                {"type": "http.response.body", "body": b"one", "more_body": True}
            )
            events.append("sent-one")
            await send(
                {"type": "http.response.body", "body": b"two", "more_body": False}
            )

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body") == b"one":
                events.append("got-one")

        asyncio.run(
            PWAMiddleware(downstream)(
                {"type": "http", "method": "GET", "path": "/gradio_api/queue/data", "headers": []},
                receive,
                send,
            )
        )
        self.assertEqual(events, ["got-one", "sent-one"])


class IconTests(unittest.TestCase):
    def test_icon_sizes(self):
        expected = {
            "icon-192.png": (192, 192),
            "icon-512.png": (512, 512),
            "icon-maskable-512.png": (512, 512),
            "apple-touch-icon.png": (180, 180),
        }
        for name, size in expected.items():
            with self.subTest(name=name):
                with Image.open(PWA_DIR / name) as image:
                    self.assertEqual(image.size, size)
                    self.assertEqual(image.mode, "RGB")
        self.assertEqual(MANIFEST["icons"][0]["sizes"], "192x192")
        self.assertEqual(MANIFEST["icons"][2]["purpose"], "maskable")


class BundleTests(unittest.TestCase):
    def test_mac_app_copies_pwa_into_the_bundle(self):
        script = Path(__file__).resolve().parents[1] / "macos" / "build_app.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn("pwa.py", text)
        self.assertIn('"$ROOT/pwa/"*', text)


class LaunchTests(unittest.TestCase):
    def test_launch_kwargs_keep_desktop_on_localhost_and_enable_pwa(self):
        kwargs = launch_kwargs(prevent_thread_lock=True, inbrowser=False)
        self.assertEqual(kwargs["server_name"], "127.0.0.1")
        self.assertTrue(kwargs["pwa"])
        self.assertTrue(kwargs["favicon_path"].endswith("icon-192.png"))
        classes = [item.cls for item in kwargs["app_kwargs"]["middleware"]]
        self.assertIn(PWAMiddleware, classes)
        again = launch_kwargs(app_kwargs=kwargs["app_kwargs"])
        classes = [item.cls for item in again["app_kwargs"]["middleware"]]
        self.assertEqual(classes.count(PWAMiddleware), 1)

    def test_phone_hint_does_not_call_this_an_app_store_build(self):
        local = phone_hint("127.0.0.1", 7860)
        self.assertIn("127.0.0.1:7860", local)
        self.assertIn("--host 0.0.0.0", local)
        shared = phone_hint("0.0.0.0", 7860)
        self.assertIn("Agregar a pantalla de inicio", shared)
        self.assertIn("no es una app de la App Store", shared)
        self.assertNotIn("IPA", shared)

    def test_gradio_serves_install_tags(self):
        from app import build_server

        port = _free_port()
        demo = build_server()
        demo.launch(
            **launch_kwargs(
                server_name="127.0.0.1",
                server_port=port,
                prevent_thread_lock=True,
                quiet=True,
            )
        )
        try:
            page = _wait(f"http://127.0.0.1:{port}/")
            self.assertIn("apple-mobile-web-app-capable", page.text)
            self.assertIn("apple-touch-icon", page.text)
            self.assertIn('rel="manifest"', page.text)
            self.assertIn("viewport-fit=cover", page.text)
            manifest = httpx.get(f"http://127.0.0.1:{port}/manifest.json", timeout=10)
            self.assertEqual(manifest.status_code, 200)
            payload = manifest.json()
            self.assertEqual(payload["display"], "standalone")
            self.assertEqual(payload["short_name"], APP_SHORT_NAME)
            self.assertEqual(payload["theme_color"], THEME_COLOR)
            icon = httpx.get(
                f"http://127.0.0.1:{port}/pwa/icon-512.png", timeout=10
            )
            self.assertEqual(icon.status_code, 200)
            self.assertTrue(icon.headers["content-type"].startswith("image/png"))
        finally:
            demo.close()


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait(url, timeout=60):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=3)
            if response.status_code == 200 and "apple-mobile-web-app-capable" in response.text:
                return response
            last = response.status_code
        except httpx.HTTPError as exc:
            last = exc
        time.sleep(0.3)
    raise AssertionError(f"el servidor no quedó listo: {last}")


if __name__ == "__main__":
    unittest.main()
