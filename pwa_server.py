"""PWA files for iOS Safari Add to Home Screen.

Gradio applies launch(head=...) in the browser after load. Safari reads the
initial HTML for apple-mobile-web-app-* and the touch icon, so those tags are
inserted into the document before it is sent. Icons and the service worker
are static and carry no audio. The manifest stays token-free unless the
request already presented the access token; then start_url keeps the home
screen icon signed in.
"""
import json
import mimetypes
import os

from fastapi.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
PWA_DIR = os.path.join(HERE, "pwa")
FAVICON_PATH = os.path.join(PWA_DIR, "icon-192.png")
_HEAD_PATH = os.path.join(PWA_DIR, "head.html")
_MANIFEST = "href=\"/pwa/manifest.webmanifest\""

mimetypes.add_type("application/manifest+json", ".webmanifest")


def pwa_head():
    with open(_HEAD_PATH, encoding="utf-8") as handle:
        return handle.read().strip()


def inject_pwa_html(html):
    """Point the manifest at our file and add the Apple tags once."""
    html = html.replace('href="/manifest.json"', _MANIFEST)
    snippet = pwa_head()
    if "apple-mobile-web-app-capable" not in html and "</head>" in html:
        html = html.replace("</head>", snippet + "\n</head>", 1)
    return html


class InjectPWAHead:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "GET":
            await self.app(scope, receive, send)
            return
        if scope.get("path") != "/":
            await self.app(scope, receive, send)
            return

        passthrough = False
        held = None
        chunks = []

        async def send_wrapper(message):
            nonlocal passthrough, held
            kind = message["type"]
            if kind == "http.response.start":
                content_type = _header(message.get("headers") or [], b"content-type")
                if b"text/html" not in content_type:
                    passthrough = True
                    await send(message)
                    return
                held = message
                return
            if kind != "http.response.body" or passthrough:
                await send(message)
                return
            chunks.append(message.get("body") or b"")
            if message.get("more_body"):
                return
            body = inject_pwa_html(b"".join(chunks).decode("utf-8")).encode("utf-8")
            headers = [
                (key, value)
                for key, value in (held.get("headers") or [])
                if key.lower() not in {b"content-length", b"transfer-encoding"}
            ]
            headers.append((b"content-length", str(len(body)).encode("ascii")))
            await send({
                "type": "http.response.start",
                "status": held["status"],
                "headers": headers,
            })
            await send({"type": "http.response.body", "body": body})

        await self.app(scope, receive, send_wrapper)


def manifest_response(request):
    """Static manifest. Adds the token to start_url only for an already-valid request."""
    with open(os.path.join(PWA_DIR, "manifest.webmanifest"), encoding="utf-8") as handle:
        data = json.load(handle)
    import app as audio_app

    if audio_app.auth_dependency(request) == "local":
        token = os.environ.get(audio_app.TOKEN_ENV, "")
        data["start_url"] = f"/?{audio_app.TOKEN_QUERY}={token}"
    return JSONResponse(
        data,
        media_type="application/manifest+json",
        headers={"Cache-Control": "no-store"},
    )


def install(app):
    """Serve /pwa, /sw.js, and the touch icon, and tag the HTML response."""
    if getattr(app, "_audio_separator_pwa", False):
        return
    app._audio_separator_pwa = True
    icon = os.path.join(PWA_DIR, "icon-180.png")
    worker = os.path.join(PWA_DIR, "sw.js")

    def apple_touch(_request):
        return FileResponse(icon, media_type="image/png")

    def service_worker(_request):
        return FileResponse(
            worker,
            media_type="application/javascript",
            headers={
                "Service-Worker-Allowed": "/",
                "Cache-Control": "no-cache",
            },
        )

    app.router.routes.insert(0, Mount("/pwa", app=StaticFiles(directory=PWA_DIR), name="pwa"))
    app.router.routes.insert(
        0,
        Route("/pwa/manifest.webmanifest", manifest_response, methods=["GET"]),
    )
    app.router.routes.insert(0, Route("/sw.js", service_worker, methods=["GET"]))
    app.router.routes.insert(
        0, Route("/apple-touch-icon-precomposed.png", apple_touch, methods=["GET"])
    )
    app.router.routes.insert(0, Route("/apple-touch-icon.png", apple_touch, methods=["GET"]))
    app.add_middleware(InjectPWAHead)


def launch_with_pwa(demo, kwargs):
    """demo.launch(), with PWA routes installed on the Gradio app."""
    import gradio.routes as routes

    original = routes.App.create_app

    def create_app(blocks, *args, **kw):
        app = original(blocks, *args, **kw)
        if blocks is demo:
            install(app)
        return app

    routes.App.create_app = create_app
    try:
        return demo.launch(**kwargs)
    finally:
        routes.App.create_app = original


def _header(headers, name):
    for key, value in headers:
        if key.lower() == name:
            return value
    return b""
