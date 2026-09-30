#!/usr/bin/env python3
"""Interfaz web local para analizar las donaciones de Ceneka.

    python3 donaciones_web.py
    python3 donaciones_web.py --desde donaciones.json
    python3 donaciones_web.py --puerto 8765
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import urllib.parse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import donaciones_ceneka as core

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
DEFAULT_PORT = 8765
CACHE_CANDIDATES = (
    ROOT / "donaciones.json",
    Path("/tmp/donaciones-todas.json"),
)


class Store:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.donations: list[dict] = []
        self.loading = False
        self.progress = 0
        self.error: str | None = None
        self.source = ""

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "cantidad": len(self.donations),
                "loading": self.loading,
                "progress": self.progress,
                "error": self.error,
                "source": self.source,
            }

    def set_donations(self, donations: list[dict], source: str) -> None:
        with self.lock:
            self.donations = donations
            self.source = source
            self.loading = False
            self.progress = len(donations)
            self.error = None

    def begin_load(self) -> bool:
        with self.lock:
            if self.loading:
                return False
            self.loading = True
            self.progress = 0
            self.error = None
            return True

    def fail(self, message: str) -> None:
        with self.lock:
            self.loading = False
            self.error = message


STORE = Store()


def parse_float(raw: str | None):
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_int(raw: str | None, default: int) -> int:
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def filters_from_query(query: dict) -> dict:
    return {
        "donante": (query.get("donante") or "").strip() or None,
        "fecha": (query.get("fecha") or "").strip() or None,
        "minimo": parse_float(query.get("min")),
        "maximo": parse_float(query.get("max")),
    }


def chart_rows(items: list[dict], label_key: str, count_key: str, sum_key: str) -> list[dict]:
    peak = max((item.get(count_key) or 0) for item in items) if items else 0
    rows = []
    for item in items:
        count = item.get(count_key) or 0
        amount = item.get(sum_key) or 0
        rows.append({
            "etiqueta": item.get(label_key) or "",
            "cantidad": count,
            "suma": amount,
            "suma_txt": core.format_amount(amount),
            "pct": (count / peak * 100) if peak else 0,
        })
    return rows


def public_report(donations: list[dict], top: int, offset: int, limit: int) -> dict:
    report = core.analyze(donations, top=top)
    science = report.get("ciencia") or {}
    conc = report.get("concentracion") or {}
    page = donations[offset:offset + limit]
    return {
        "cantidad": report["cantidad"],
        "kpis": [
            {"label": "Donaciones", "value": core._fmt_n(report["cantidad"])},
            {"label": "Suma", "value": core.format_amount(report["suma"])},
            {"label": "Mediana", "value": core.format_amount(report["mediana"])},
            {"label": "Promedio", "value": core.format_amount(report["promedio"])},
            {"label": "Donantes", "value": core._fmt_n(report["donantes"])},
            {"label": "Gini donantes", "value": f"{conc.get('gini_donantes', 0):.3f}"},
        ],
        "detalle": {
            "minimo": core.format_amount(report["minimo"]),
            "maximo": core.format_amount(report["maximo"]),
            "desvio": core.format_amount(report.get("desvio") or 0),
            "recurrentes": core._fmt_n(report["recurrentes"]),
            "una_vez": core._fmt_n(report["una_vez"]),
            "privadas": core._fmt_n(report["privadas"]),
            "con_link": core._fmt_n(report["con_link"]),
            "p10": core.format_amount(science.get("percentiles", {}).get("p10", 0)),
            "p90": core.format_amount(science.get("percentiles", {}).get("p90", 0)),
            "p99": core.format_amount(science.get("percentiles", {}).get("p99", 0)),
            "outliers": core._fmt_n(science.get("outliers_iqr") or 0),
            "pareto": core._pareto_line(conc.get("pareto_80_donantes") or {}, "donantes"),
        },
        "insights": core.insights_list(report),
        "rangos": chart_rows(report["por_rango"], "etiqueta", "cantidad", "suma"),
        "periodos": chart_rows(report["por_periodo"], "periodo", "cantidad", "suma"),
        "frecuencia": chart_rows(science.get("frecuencia_donante") or [], "etiqueta", "donantes", "suma"),
        "top_monto": [
            {"nombre": item["nombre"], "cantidad": item["cantidad"], "suma": core.format_amount(item["suma"])}
            for item in report["por_usuario_monto"]
        ],
        "top_cantidad": [
            {"nombre": item["nombre"], "cantidad": item["cantidad"], "suma": core.format_amount(item["suma"])}
            for item in report["por_usuario_cantidad"]
        ],
        "mayores": [
            {
                "nombre": item["nombre"],
                "fecha": item["fecha"],
                "monto": item["monto"],
                "mensaje": item.get("mensaje") or "",
            }
            for item in report["mayores"]
        ],
        "tabla": {
            "total": len(donations),
            "offset": offset,
            "items": [
                {
                    "nombre": item["nombre"],
                    "fecha": item["fecha"],
                    "monto": item["monto"],
                    "mensaje": item.get("mensaje") or "",
                }
                for item in page
            ],
        },
    }


def load_local_if_present() -> None:
    for path in CACHE_CANDIDATES:
        if path.is_file():
            STORE.set_donations(core.load_donations(str(path)), str(path))
            return


def fetch_remote(user: str) -> None:
    if not STORE.begin_load():
        return

    def on_progress(total: int) -> None:
        with STORE.lock:
            STORE.progress = total

    try:
        donations = core.fetch_latest(user, None, on_progress=on_progress)
        STORE.set_donations(donations, f"https://ceneka.net/{user}")
        cache = ROOT / "donaciones.json"
        cache.write_text(json.dumps(donations, ensure_ascii=False), encoding="utf-8")
    except Exception as error:  # noqa: BLE001 — shown in the UI
        STORE.fail(str(error))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, page_user: str = core.DEFAULT_USER, **kwargs):
        self.page_user = page_user
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(format, *args)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            return self._send_file(WEB_DIR / "donaciones.html", "text/html; charset=utf-8")
        if parsed.path == "/api/estado":
            return self._send_json(STORE.snapshot())
        if parsed.path == "/api/reporte":
            return self._send_json(self._reporte(parsed.query))
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/cargar":
            threading.Thread(target=fetch_remote, args=(self.page_user,), daemon=True).start()
            return self._send_json({"ok": True, **STORE.snapshot()})
        self.send_error(404)

    def _reporte(self, raw_query: str) -> dict:
        query = dict(urllib.parse.parse_qsl(raw_query, keep_blank_values=True))
        filtros = filters_from_query(query)
        meta = STORE.snapshot()
        with STORE.lock:
            donations = list(STORE.donations)
        filtered = core.filter_donations(
            donations,
            donante=filtros["donante"],
            fecha=filtros["fecha"],
            minimo=filtros["minimo"],
            maximo=filtros["maximo"],
        )
        top = max(1, parse_int(query.get("top"), 12))
        offset = max(0, parse_int(query.get("offset"), 0))
        limit = min(100, max(1, parse_int(query.get("limit"), 40)))
        payload = public_report(filtered, top=top, offset=offset, limit=limit)
        payload["estado"] = meta
        payload["filtros"] = {key: value for key, value in filtros.items() if value is not None}
        payload["usuario"] = self.page_user
        return payload

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Interfaz web para las donaciones de Ceneka.")
    parser.add_argument("--puerto", type=int, default=DEFAULT_PORT)
    parser.add_argument("--desde", help="JSON local para precargar")
    parser.add_argument("-u", "--usuario", default=core.DEFAULT_USER)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.desde:
        STORE.set_donations(core.load_donations(args.desde), args.desde)
    else:
        load_local_if_present()
    handler = partial(Handler, page_user=args.usuario.strip() or core.DEFAULT_USER)
    server = ThreadingHTTPServer((args.host, args.puerto), handler)
    print(f"Abrí http://{args.host}:{args.puerto}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
