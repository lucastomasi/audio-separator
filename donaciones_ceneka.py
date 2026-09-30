#!/usr/bin/env python3
"""Muestra las últimas donaciones de https://ceneka.net/losherederosdealberdi.

Usa solo la biblioteca estándar, así corre en la terminal de Mac con python3:

    python3 donaciones_ceneka.py
    python3 donaciones_ceneka.py -n 20
    python3 donaciones_ceneka.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from html import unescape

PAGE_URL = "https://ceneka.net/losherederosdealberdi"
API_URL = "https://ceneka.net/mp/apis/listarDonaciones.php"
DEFAULT_USER = "losherederosdealberdi"
PAGE_SIZE_MAX = 50
DEFAULT_COUNT = 10


def format_amount(valor) -> str:
    """Formatea el monto como lo muestra Ceneka: '$ 8 000'."""
    try:
        number = float(valor)
    except (TypeError, ValueError):
        text = "" if valor is None else str(valor).strip()
        return f"$ {text}" if text else "$ —"
    if number.is_integer():
        whole = f"{int(number):,}".replace(",", " ")
        return f"$ {whole}"
    formatted = f"{number:,.2f}".replace(",", " ")
    return f"$ {formatted}"


def normalize_donation(raw: dict) -> dict:
    mensaje = unescape(str(raw.get("mensaje") or "")).strip()
    nombre = unescape(str(raw.get("nombre") or "")).strip() or "Anónimo"
    return {
        "id": raw.get("id"),
        "monto": format_amount(raw.get("valor")),
        "valor": raw.get("valor"),
        "fecha": str(raw.get("fecha") or "").strip(),
        "nombre": nombre,
        "mensaje": mensaje,
        "privada": bool(raw.get("privado")),
    }


def fetch_page(user: str, page: int, limit: int, timeout: float = 20) -> list:
    query = urllib.parse.urlencode({"u": user, "p": page, "limit": limit})
    request = urllib.request.Request(
        f"{API_URL}?{query}",
        headers={
            "User-Agent": "donaciones-ceneka/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("Ceneka no devolvió una lista de donaciones.")
    return payload


def fetch_latest(user: str, count: int, fetcher=fetch_page) -> list[dict]:
    if count < 1:
        return []
    page_size = min(PAGE_SIZE_MAX, count)
    collected: list[dict] = []
    page = 0
    while len(collected) < count:
        chunk = fetcher(user, page, page_size)
        if not chunk:
            break
        collected.extend(normalize_donation(item) for item in chunk)
        if len(chunk) < page_size:
            break
        page += 1
    return collected[:count]


def render_text(donations: list[dict], user: str) -> str:
    lines = [
        f"Últimas {len(donations)} donaciones — {user}",
        PAGE_URL if user.lower() == DEFAULT_USER else f"https://ceneka.net/{user}",
        "",
    ]
    if not donations:
        lines.append("No hay donaciones para mostrar.")
        return "\n".join(lines)
    for index, donation in enumerate(donations, start=1):
        privada = "  (privada)" if donation["privada"] else ""
        lines.append(f"{index:>2}. {donation['monto']:<12} {donation['nombre']}{privada}")
        if donation["fecha"]:
            lines.append(f"    {donation['fecha']}")
        if donation["mensaje"]:
            lines.append(f"    {donation['mensaje']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lista las últimas donaciones públicas de una página de Ceneka."
    )
    parser.add_argument(
        "-n",
        "--cantidad",
        type=int,
        default=DEFAULT_COUNT,
        help=f"cuántas donaciones traer (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "-u",
        "--usuario",
        default=DEFAULT_USER,
        help=f"slug de la página (default: {DEFAULT_USER})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="imprimir JSON en lugar de texto",
    )
    args = parser.parse_args(argv)
    if args.cantidad < 1:
        parser.error("--cantidad tiene que ser 1 o más")
    if not args.usuario.strip():
        parser.error("--usuario no puede estar vacío")
    args.usuario = args.usuario.strip()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        donations = fetch_latest(args.usuario, args.cantidad)
    except urllib.error.URLError as error:
        print(f"No se pudo conectar con Ceneka: {error.reason}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, RuntimeError, TimeoutError, OSError) as error:
        print(f"No se pudieron leer las donaciones: {error}", file=sys.stderr)
        return 1
    if args.json:
        json.dump(donations, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_text(donations, args.usuario))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
