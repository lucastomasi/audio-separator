#!/usr/bin/env python3
"""Muestra las donaciones de https://ceneka.net/losherederosdealberdi.

Usa solo la biblioteca estándar, así corre en la terminal de Mac con python3.
Sin -n recorre todas las páginas (el listado público son unas decenas de miles):

    python3 donaciones_ceneka.py
    python3 donaciones_ceneka.py -n 20
    python3 donaciones_ceneka.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html import unescape

PAGE_URL = "https://ceneka.net/losherederosdealberdi"
API_URL = "https://ceneka.net/mp/apis/listarDonaciones.php"
DEFAULT_USER = "losherederosdealberdi"
PAGE_SIZE_MAX = 50
MAX_PAGES = 20000


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


def fetch_page(user: str, page: int, limit: int, timeout: float = 30) -> list:
    query = urllib.parse.urlencode({"u": user, "p": page, "limit": limit})
    request = urllib.request.Request(
        f"{API_URL}?{query}",
        headers={
            "User-Agent": "donaciones-ceneka/1.0",
            "Accept": "application/json",
        },
    )
    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(delay)
            delay *= 2
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt == 3:
                raise
            time.sleep(delay)
            delay *= 2
    else:
        raise RuntimeError(f"No se pudo leer la página {page}: {last_error}")
    if not isinstance(payload, list):
        raise RuntimeError("Ceneka no devolvió una lista de donaciones.")
    return payload


def fetch_latest(user: str, count: int | None = None, fetcher=fetch_page, on_progress=None) -> list[dict]:
    """Trae donaciones de la más nueva a la más vieja.

    count=None recorre el listado entero. Ceneka entrega como máximo 50 por página.
    """
    if count is not None and count < 1:
        return []
    page_size = PAGE_SIZE_MAX if count is None else min(PAGE_SIZE_MAX, count)
    collected: list[dict] = []
    seen: set = set()
    page = 0
    while count is None or len(collected) < count:
        if page >= MAX_PAGES:
            break
        chunk = fetcher(user, page, page_size)
        if not chunk:
            break
        added = 0
        for item in chunk:
            donation_id = item.get("id")
            if donation_id in seen:
                continue
            seen.add(donation_id)
            collected.append(normalize_donation(item))
            added += 1
            if count is not None and len(collected) >= count:
                break
        if on_progress is not None:
            on_progress(len(collected))
        if added == 0 or len(chunk) < page_size:
            break
        page += 1
    return collected


def render_text(donations: list[dict], user: str, todas: bool = False) -> str:
    titulo = f"Todas las donaciones ({len(donations)})" if todas else f"Últimas {len(donations)} donaciones"
    lines = [
        f"{titulo} — {user}",
        PAGE_URL if user.lower() == DEFAULT_USER else f"https://ceneka.net/{user}",
        "",
    ]
    if not donations:
        lines.append("No hay donaciones para mostrar.")
        return "\n".join(lines) + "\n"
    width = len(str(len(donations)))
    for index, donation in enumerate(donations, start=1):
        privada = "  (privada)" if donation["privada"] else ""
        lines.append(f"{index:>{width}}. {donation['monto']:<12} {donation['nombre']}{privada}")
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
        default=None,
        help="cuántas donaciones traer; si se omite, trae todas",
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
    if args.cantidad is not None and args.cantidad < 1:
        parser.error("--cantidad tiene que ser 1 o más")
    if not args.usuario.strip():
        parser.error("--usuario no puede estar vacío")
    args.usuario = args.usuario.strip()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    def on_progress(total: int) -> None:
        print(f"\rLeyendo donaciones… {total}", file=sys.stderr, end="", flush=True)

    try:
        donations = fetch_latest(args.usuario, args.cantidad, on_progress=on_progress)
    except urllib.error.URLError as error:
        print(file=sys.stderr)
        print(f"No se pudo conectar con Ceneka: {error.reason}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, RuntimeError, TimeoutError, OSError) as error:
        print(file=sys.stderr)
        print(f"No se pudieron leer las donaciones: {error}", file=sys.stderr)
        return 1
    print(file=sys.stderr)
    if args.json:
        json.dump(donations, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_text(donations, args.usuario, todas=args.cantidad is None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
