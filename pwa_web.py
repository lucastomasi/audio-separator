"""Home-screen install metadata for the Gradio UI.

Safari on iPhone reads these tags from the first HTML response. Gradio's
``head=`` argument is applied in the browser after load, so the tags are
inserted by middleware before ``</head>``.
"""
import os
import socket

from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageDraw
from starlette.datastructures import MutableHeaders
from starlette.routing import Route

ROOT = os.path.dirname(os.path.abspath(__file__))
PWA_DIR = os.path.join(ROOT, "pwa")

NAME = "Audio Separator"
SHORT_NAME = "Separator"
DESCRIPTION = (
    "Separá la voz del instrumental y convertí la voz con RVC. "
    "El iPhone abre esta página; los modelos corren en la computadora."
)
THEME_COLOR = "#4f46e5"
BACKGROUND_COLOR = "#eef2ff"
MARKER = b"audio-separator-pwa"

ICON_FILES = {
    "icon-180.png": (180, False),
    "icon-192.png": (192, False),
    "icon-512.png": (512, False),
    "icon-maskable-512.png": (512, True),
    "favicon-32.png": (32, False),
}


def _rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[index : index + 2], 16) for index in (0, 2, 4))


def render_icon(size, maskable=False):
    """Square indigo icon with a white waveform. iOS masks the corners itself."""
    top = _rgb("#6366f1")
    bottom = _rgb("#312e81")
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    span = max(size - 1, 1)
    for y in range(size):
        blend = y / span
        color = tuple(int(top[channel] + (bottom[channel] - top[channel]) * blend) for channel in range(3))
        for x in range(size):
            pixels[x, y] = color

    pad_ratio = 0.27 if maskable else 0.18
    pad = int(size * pad_ratio)
    inner = max(size - 2 * pad, 1)
    count = 5
    gap = max(1, int(size * 0.03))
    bar_width = max(1, (inner - gap * (count - 1)) / count)
    heights = (0.42, 0.7, 1.0, 0.58, 0.84)
    max_height = inner * (0.72 if maskable else 0.78)
    center_y = size / 2
    draw = ImageDraw.Draw(image)
    for index, ratio in enumerate(heights):
        height = max(2, int(max_height * ratio))
        x0 = pad + index * (bar_width + gap)
        x1 = x0 + bar_width
        y0 = center_y - height / 2
        y1 = center_y + height / 2
        radius = max(1, int(min(bar_width, height) / 2))
        draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=(255, 255, 255))
    return image


def write_icons(directory=PWA_DIR):
    os.makedirs(directory, exist_ok=True)
    written = []
    for name, (size, maskable) in ICON_FILES.items():
        path = os.path.join(directory, name)
        render_icon(size, maskable=maskable).save(path, format="PNG")
        written.append(path)
    ico_path = os.path.join(directory, "favicon.ico")
    render_icon(32).save(ico_path, format="ICO")
    written.append(ico_path)
    return written


def _icon(filename, size, purpose):
    return {
        "src": f"/pwa/{filename}",
        "sizes": f"{size}x{size}",
        "type": "image/png",
        "purpose": purpose,
    }


def manifest():
    return {
        "name": NAME,
        "short_name": SHORT_NAME,
        "description": DESCRIPTION,
        "id": "/",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "orientation": "any",
        "background_color": BACKGROUND_COLOR,
        "theme_color": THEME_COLOR,
        "lang": "es",
        "prefer_related_applications": False,
        "icons": [
            _icon("icon-180.png", 180, "any"),
            _icon("icon-192.png", 192, "any"),
            _icon("icon-512.png", 512, "any"),
            _icon("icon-maskable-512.png", 512, "maskable"),
        ],
    }


def head_markup():
    """Tags Safari needs in the original HTML to Add to Home Screen."""
    return f"""<!-- {MARKER.decode()} -->
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="{SHORT_NAME}">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="application-name" content="{NAME}">
<meta name="theme-color" content="{THEME_COLOR}">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="icon" href="/pwa/favicon-32.png" type="image/png" sizes="32x32">
<script>
if ("serviceWorker" in navigator) {{
  window.addEventListener("load", function () {{
    navigator.serviceWorker.register("/sw.js", {{ scope: "/" }}).catch(function () {{}});
  }});
}}
</script>
"""


def service_worker_js():
    # The fetch listener does not call respondWith. Gradio audio, uploads, and
    # the queue stay on the network. iPhone Add to Home Screen does not require
    # this file; it is here for browsers that look for a service worker.
    return """self.addEventListener("install", function (event) {
  event.waitUntil(self.skipWaiting());
});
self.addEventListener("activate", function (event) {
  event.waitUntil(self.clients.claim());
});
self.addEventListener("fetch", function () {});
"""


def inject_pwa_head(body):
    if not body or MARKER in body:
        return body
    needle = b"</head>"
    index = body.rfind(needle)
    if index < 0:
        return body
    snippet = head_markup().encode("utf-8")
    return body[:index] + snippet + body[index:]


class PwaHeadMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "GET":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or "/"
        if path not in ("/", ""):
            await self.app(scope, receive, send)
            return

        start = {}
        chunks = []

        async def capture(message):
            if message["type"] == "http.response.start":
                start.update(message)
                return
            if message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
                if message.get("more_body", False):
                    return
                await self._send(send, start, b"".join(chunks))
                return
            await send(message)

        await self.app(scope, receive, capture)

    async def _send(self, send, start, body):
        if not start:
            return
        headers = MutableHeaders(raw=[tuple(pair) for pair in start.get("headers", [])])
        content_type = headers.get("content-type", "")
        encoding = headers.get("content-encoding", "")
        if "text/html" in content_type and not encoding:
            updated = inject_pwa_head(body)
            if updated is not body:
                body = updated
                headers["content-length"] = str(len(body))
                if "cache-control" not in headers:
                    headers["cache-control"] = "no-cache"
        start["headers"] = headers.raw
        await send(start)
        await send({"type": "http.response.body", "body": body, "more_body": False})


def _png(filename):
    path = os.path.join(PWA_DIR, filename)

    async def endpoint(_request):
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

    return endpoint


def install_pwa(app):
    """Serve the manifest, icons, and service worker, and tag the Gradio HTML."""
    if getattr(app.state, "audio_separator_pwa", False):
        return app
    app.state.audio_separator_pwa = True

    async def manifest_endpoint(_request):
        return JSONResponse(
            manifest(),
            media_type="application/manifest+json",
            headers={"Cache-Control": "no-cache"},
        )

    async def service_worker(_request):
        return Response(
            service_worker_js(),
            media_type="application/javascript",
            headers={
                "Cache-Control": "no-cache",
                "Service-Worker-Allowed": "/",
            },
        )

    touch = _png("icon-180.png")
    routes = [
        Route("/manifest.json", manifest_endpoint, methods=["GET"]),
        Route("/manifest.webmanifest", manifest_endpoint, methods=["GET"]),
        Route("/sw.js", service_worker, methods=["GET"]),
        Route("/apple-touch-icon.png", touch, methods=["GET"]),
        Route("/apple-touch-icon-precomposed.png", touch, methods=["GET"]),
    ]
    for filename in ICON_FILES:
        routes.append(Route(f"/pwa/{filename}", _png(filename), methods=["GET"]))

    for route in reversed(routes):
        app.router.routes.insert(0, route)
    app.add_middleware(PwaHeadMiddleware)
    return app


def lan_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


def phone_url(host, port):
    if host in ("0.0.0.0", "::", ""):
        ip = lan_ip()
        if not ip:
            return None
        return f"http://{ip}:{port}"
    if host in ("127.0.0.1", "localhost"):
        return None
    return f"http://{host}:{port}"


def print_phone_url(host, port):
    url = phone_url(host, port)
    if not url:
        return
    print(f"* En el iPhone, abrí Safari en {url}")
    print("* Después: Compartir → Agregar a pantalla de inicio. No es una app de la App Store.")
