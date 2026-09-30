#!/usr/bin/env python3
"""Interfaz web local para analizar las donaciones de Ceneka.

    python3 donaciones_web.py
    python3 donaciones_web.py --puerto 8765

El grupo completo (~40 mil) viene en donaciones.json.gz y se abre solo.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import threading
import urllib.parse
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import donaciones_ceneka as core

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
DEFAULT_PORT = 8765
BUNDLED = ROOT / "donaciones.json.gz"
CACHE_CANDIDATES = (
    BUNDLED,
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
        packed = json.dumps(donations, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        BUNDLED.write_bytes(gzip.compress(packed, compresslevel=9))
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
        if parsed.path == "/listado":
            return self._send_bytes(self._listado_html(parsed.query), "text/html; charset=utf-8")
        if parsed.path == "/api/listado.csv":
            return self._send_bytes(self._listado_csv(parsed.query), "text/csv; charset=utf-8", "listado-donaciones.csv")
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/cargar":
            threading.Thread(target=fetch_remote, args=(self.page_user,), daemon=True).start()
            return self._send_json({"ok": True, **STORE.snapshot()})
        self.send_error(404)

    def _filtered(self, raw_query: str) -> tuple[list[dict], dict]:
        query = dict(urllib.parse.parse_qsl(raw_query, keep_blank_values=True))
        filtros = filters_from_query(query)
        with STORE.lock:
            donations = list(STORE.donations)
        filtered = core.filter_donations(
            donations,
            donante=filtros["donante"],
            fecha=filtros["fecha"],
            minimo=filtros["minimo"],
            maximo=filtros["maximo"],
        )
        return filtered, filtros

    def _reporte(self, raw_query: str) -> dict:
        query = dict(urllib.parse.parse_qsl(raw_query, keep_blank_values=True))
        filtered, filtros = self._filtered(raw_query)
        top = max(1, parse_int(query.get("top"), 12))
        payload = public_report(filtered, top=top, offset=0, limit=0)
        payload["estado"] = STORE.snapshot()
        payload["filtros"] = {key: value for key, value in filtros.items() if value is not None}
        payload["usuario"] = self.page_user
        return payload

    def _listado_html(self, raw_query: str) -> bytes:
        filtered, _ = self._filtered(raw_query)
        rows = []
        for index, item in enumerate(filtered, start=1):
            rows.append(
                "<tr>"
                f"<td>{index}</td>"
                f"<td>{escape(item['nombre'])}</td>"
                f"<td>{escape(item['fecha'])}</td>"
                f"<td class=\"monto\">{escape(item['monto'])}</td>"
                f"<td class=\"msg\">{escape(item.get('mensaje') or '')}</td>"
                "</tr>"
            )
        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Listado completo · {escape(self.page_user)}</title>
  <link rel="stylesheet" href="/donaciones.css">
</head>
<body>
  <p class="eyebrow"><a href="/">← Análisis</a></p>
  <h1>Listado completo</h1>
  <p class="status">{len(filtered)} donaciones. Cada fila es una donación; no se junta ni se borra nada.</p>
  <p><a href="/api/listado.csv?{escape(raw_query, quote=True)}">Descargar CSV</a></p>
  <div class="table-wrap">
    <table>
      <thead><tr><th>#</th><th>Usuario</th><th>Fecha</th><th>Monto</th><th>Mensaje</th></tr></thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
        return html.encode("utf-8")

    def _listado_csv(self, raw_query: str) -> bytes:
        filtered, _ = self._filtered(raw_query)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["n", "id", "usuario", "fecha", "monto", "valor", "mensaje"])
        for index, item in enumerate(filtered, start=1):
            writer.writerow([
                index,
                item.get("id"),
                item.get("nombre"),
                item.get("fecha"),
                item.get("monto"),
                item.get("valor"),
                item.get("mensaje") or "",
            ])
        return buffer.getvalue().encode("utf-8")

    def _send_file(self, path: Path, content_type: str) -> None:
        self._send_bytes(path.read_bytes(), content_type)

    def _send_bytes(self, data: bytes, content_type: str, download: str | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
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
