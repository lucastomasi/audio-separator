"""PWA shell so iPhone Safari can Add to Home Screen.

Gradio puts a manifest link in the first HTML response, but custom `head`
tags are inserted only after JavaScript runs. Safari reads the first response
when it builds the home-screen icon, so the Apple meta tags and icons are
applied here, before the page reaches the browser.
"""

from __future__ import annotations

import re
from pathlib import Path

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import FileResponse, JSONResponse, Response

ROOT = Path(__file__).resolve().parent
PWA_DIR = ROOT / "pwa"

APP_NAME = "Audio Separator"
APP_SHORT_NAME = "Separator"
THEME_COLOR = "#0f172a"
BACKGROUND_COLOR = "#f8fafc"
MARKER = "<!-- audio-separator-pwa -->"

ICON_FILES = {
    "/apple-touch-icon.png": PWA_DIR / "apple-touch-icon.png",
    "/apple-touch-icon-precomposed.png": PWA_DIR / "apple-touch-icon.png",
    "/pwa/apple-touch-icon.png": PWA_DIR / "apple-touch-icon.png",
    "/pwa/icon-192.png": PWA_DIR / "icon-192.png",
    "/pwa/icon-512.png": PWA_DIR / "icon-512.png",
    "/pwa/icon-maskable-512.png": PWA_DIR / "icon-maskable-512.png",
    "/sw.js": PWA_DIR / "sw.js",
}

MANIFEST_PATHS = {"/manifest.json", "/manifest.webmanifest"}

MANIFEST = {
    "id": "/",
    "name": APP_NAME,
    "short_name": APP_SHORT_NAME,
    "description": (
        "Separá la voz del instrumental desde el navegador. "
        "El procesamiento corre en la computadora, no en el teléfono."
    ),
    "lang": "es",
    "dir": "ltr",
    "start_url": "/",
    "scope": "/",
    "display": "standalone",
    "orientation": "any",
    "background_color": BACKGROUND_COLOR,
    "theme_color": THEME_COLOR,
    "prefer_related_applications": False,
    "icons": [
        {
            "src": "/pwa/icon-192.png",
            "sizes": "192x192",
            "type": "image/png",
            "purpose": "any",
        },
        {
            "src": "/pwa/icon-512.png",
            "sizes": "512x512",
            "type": "image/png",
            "purpose": "any",
        },
        {
            "src": "/pwa/icon-maskable-512.png",
            "sizes": "512x512",
            "type": "image/png",
            "purpose": "maskable",
        },
    ],
}

HEAD_SNIPPET = f"""{MARKER}
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="{APP_SHORT_NAME}">
<meta name="application-name" content="{APP_NAME}">
<meta name="theme-color" content="{THEME_COLOR}">
<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="192x192" href="/pwa/icon-192.png">
<script>
if ("serviceWorker" in navigator) {{
  window.addEventListener("load", function () {{
    navigator.serviceWorker.register("/sw.js").catch(function () {{}});
  }});
}}
</script>
"""

_META_TAG = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_VIEWPORT_NAME = re.compile(r"""name\s*=\s*["']viewport["']""", re.IGNORECASE)
_CONTENT_ATTR = re.compile(
    r"""(content\s*=\s*["'])([^"']*)(["'])""", re.IGNORECASE
)


def favicon_path() -> str:
    return str(PWA_DIR / "icon-192.png")


def _add_viewport_fit(html: str) -> str:
    def repl(match: re.Match[str]) -> str:
        tag = match.group(0)
        if not _VIEWPORT_NAME.search(tag) or "viewport-fit" in tag:
            return tag

        def content(attr: re.Match[str]) -> str:
            value = attr.group(2).rstrip().rstrip(",")
            return f"{attr.group(1)}{value}, viewport-fit=cover{attr.group(3)}"

        return _CONTENT_ATTR.sub(content, tag, count=1)

    return _META_TAG.sub(repl, html)


def inject_html(html: str) -> str:
    """Insert Apple tags into the first HTML response. Idempotent."""
    if MARKER in html:
        return html
    html = _add_viewport_fit(html)
    lower = html.lower()
    index = lower.rfind("</head>")
    if index == -1:
        return html
    return html[:index] + HEAD_SNIPPET + html[index:]


def _pwa_response(path: str) -> Response:
    if path in MANIFEST_PATHS:
        return JSONResponse(
            MANIFEST,
            media_type="application/manifest+json",
            headers={"Cache-Control": "no-cache"},
        )
    file_path = ICON_FILES[path]
    if file_path.suffix == ".js":
        media_type = "application/javascript"
    else:
        media_type = "image/png"
    return FileResponse(
        file_path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


class PWAMiddleware:
    """Serve the manifest and icons, and tag the HTML shell.

    Streaming responses (the Gradio queue) are forwarded untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        path = scope.get("path", "")
        if method in ("GET", "HEAD") and (
            path in ICON_FILES or path in MANIFEST_PATHS
        ):
            await _pwa_response(path)(scope, receive, send)
            return
        if method not in ("GET", "HEAD"):
            await self.app(scope, receive, send)
            return

        state: dict = {"html": False, "start": None, "body": bytearray()}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = Headers(raw=message.get("headers", []))
                if "text/html" in headers.get("content-type", "").lower():
                    state["html"] = True
                    state["start"] = message
                    return
                await send(message)
                return
            if message["type"] == "http.response.body" and state["html"]:
                state["body"].extend(message.get("body") or b"")
                if message.get("more_body"):
                    return
                text = inject_html(state["body"].decode("utf-8", errors="replace"))
                data = text.encode("utf-8")
                start = state["start"]
                headers = MutableHeaders(raw=list(start["headers"]))
                headers["content-length"] = str(len(data))
                if "content-encoding" in headers:
                    del headers["content-encoding"]
                await send(
                    {
                        "type": "http.response.start",
                        "status": start["status"],
                        "headers": headers.raw,
                    }
                )
                await send(
                    {"type": "http.response.body", "body": data, "more_body": False}
                )
                return
            await send(message)

        await self.app(scope, receive, send_wrapper)
