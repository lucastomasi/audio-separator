#!/usr/bin/env python3
"""Muestra y analiza las donaciones de https://ceneka.net/losherederosdealberdi.

Usa solo la biblioteca estándar, así corre en la terminal de Mac con python3.
Sin -n recorre todas las páginas (el listado público son unas decenas de miles):

    python3 donaciones_ceneka.py --metricas
    python3 donaciones_ceneka.py --metricas --desde donaciones.json
    python3 donaciones_ceneka.py --metricas --donante Disociandri --min 500
    python3 donaciones_ceneka.py --tabla --fecha hoy
    python3 donaciones_ceneka.py -n 20
    python3 donaciones_ceneka.py --json --guardar donaciones.json
"""

from __future__ import annotations

import argparse
import json
import re
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
METRICAS_TOP = 15

_FECHA_RELATIVA = re.compile(
    r"hace\s+(\d+|un|una)\s+(segundo|minuto|hora|día|dia|semana|mes|año|ano)s?",
    re.IGNORECASE,
)
_UNIDAD_SEGUNDOS = {
    "segundo": 1,
    "minuto": 60,
    "hora": 3600,
    "día": 86400,
    "dia": 86400,
    "semana": 604800,
    "mes": 30 * 86400,
    "año": 365 * 86400,
    "ano": 365 * 86400,
}
_PERIODOS = {
    "hoy": 86400,
    "dia": 86400,
    "día": 86400,
    "semana": 604800,
    "mes": 30 * 86400,
    "año": 365 * 86400,
    "ano": 365 * 86400,
}
_RANGOS_MONTO = (
    (0, 100, "Hasta $ 100"),
    (100, 500, "$ 100 – $ 499"),
    (500, 1000, "$ 500 – $ 999"),
    (1000, 5000, "$ 1 000 – $ 4 999"),
    (5000, 10000, "$ 5 000 – $ 9 999"),
    (10000, 50000, "$ 10 000 – $ 49 999"),
    (50000, None, "$ 50 000 o más"),
)
_BUCKET_ORDEN = ("Hoy", "Esta semana", "Este mes", "Este año", "Años anteriores")


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


def donation_value(donation: dict) -> float:
    try:
        return float(donation.get("valor") or 0)
    except (TypeError, ValueError):
        return 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def fecha_age_seconds(fecha: str) -> int | None:
    text = (fecha or "").strip()
    if not text:
        return None
    match = _FECHA_RELATIVA.search(text)
    if not match:
        return None
    raw_n, unit = match.group(1), match.group(2).lower()
    count = 1 if raw_n.lower() in {"un", "una"} else int(raw_n)
    return count * _UNIDAD_SEGUNDOS[unit]


def fecha_bucket(fecha: str) -> str:
    age = fecha_age_seconds(fecha)
    if age is None:
        return fecha.strip() or "(sin fecha)"
    if age < 86400:
        return "Hoy"
    if age < 604800:
        return "Esta semana"
    if age < 30 * 86400:
        return "Este mes"
    if age < 365 * 86400:
        return "Este año"
    return "Años anteriores"


def matches_fecha(fecha: str, filtro: str) -> bool:
    key = filtro.strip().lower()
    if not key:
        return True
    if key in _PERIODOS:
        age = fecha_age_seconds(fecha)
        return age is not None and age <= _PERIODOS[key]
    return key in (fecha or "").lower()


def normalize_donation(raw: dict) -> dict:
    mensaje = unescape(str(raw.get("mensaje") or "")).strip()
    nombre = unescape(str(raw.get("nombre") or "")).strip() or "Anónimo"
    valor = raw.get("valor")
    return {
        "id": raw.get("id"),
        "monto": raw.get("monto") if raw.get("monto") and valor is not None and str(raw.get("monto")).startswith("$") else format_amount(valor),
        "valor": valor,
        "fecha": str(raw.get("fecha") or "").strip(),
        "nombre": nombre,
        "mensaje": mensaje,
        "privada": bool(raw.get("privado") if "privado" in raw else raw.get("privada")),
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


def load_donations(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("donaciones") or payload.get("donations") or []
    if not isinstance(payload, list):
        raise RuntimeError(f"{path} no tiene una lista de donaciones.")
    return [normalize_donation(item) for item in payload]


def filter_donations(
    donations: list[dict],
    donante: str | None = None,
    fecha: str | None = None,
    minimo: float | None = None,
    maximo: float | None = None,
) -> list[dict]:
    needle = donante.strip().lower() if donante else ""
    filtered = []
    for donation in donations:
        if needle and needle not in donation["nombre"].lower():
            continue
        if fecha and not matches_fecha(donation["fecha"], fecha):
            continue
        amount = donation_value(donation)
        if minimo is not None and amount < minimo:
            continue
        if maximo is not None and amount > maximo:
            continue
        filtered.append(donation)
    return filtered


def analyze(donations: list[dict], top: int = METRICAS_TOP) -> dict:
    values = [donation_value(item) for item in donations]
    total = sum(values)
    by_user: dict[str, dict] = {}
    by_fecha: dict[str, dict] = {}
    by_bucket: dict[str, dict] = {name: {"cantidad": 0, "suma": 0.0} for name in _BUCKET_ORDEN}
    rangos = [{"etiqueta": label, "cantidad": 0, "suma": 0.0} for _, _, label in _RANGOS_MONTO]
    con_mensaje = 0
    privadas = 0
    con_link = 0
    for donation, amount in zip(donations, values):
        nombre = donation["nombre"]
        user = by_user.setdefault(nombre, {"nombre": nombre, "cantidad": 0, "suma": 0.0, "maximo": 0.0})
        user["cantidad"] += 1
        user["suma"] += amount
        if amount > user["maximo"]:
            user["maximo"] = amount
        fecha = donation["fecha"] or "(sin fecha)"
        periodo = by_fecha.setdefault(fecha, {"fecha": fecha, "cantidad": 0, "suma": 0.0})
        periodo["cantidad"] += 1
        periodo["suma"] += amount
        bucket = fecha_bucket(fecha)
        slot = by_bucket.setdefault(bucket, {"cantidad": 0, "suma": 0.0})
        slot["cantidad"] += 1
        slot["suma"] += amount
        for index, (low, high, _) in enumerate(_RANGOS_MONTO):
            if amount >= low and (high is None or amount < high):
                rangos[index]["cantidad"] += 1
                rangos[index]["suma"] += amount
                break
        if donation.get("mensaje"):
            con_mensaje += 1
        if donation.get("privada"):
            privadas += 1
        if "http://" in donation.get("mensaje", "") or "https://" in donation.get("mensaje", ""):
            con_link += 1

    users = list(by_user.values())
    recurrentes = [item for item in users if item["cantidad"] >= 2]
    una_vez = len(users) - len(recurrentes)
    ordered_values = sorted(values, reverse=True)
    def share(count: int) -> float:
        if not ordered_values or count <= 0:
            return 0.0
        return sum(ordered_values[:count]) / total * 100 if total else 0.0

    top_n = max(1, top)
    return {
        "cantidad": len(donations),
        "suma": total,
        "promedio": total / len(values) if values else 0.0,
        "mediana": median(values),
        "minimo": min(values) if values else 0.0,
        "maximo": max(values) if values else 0.0,
        "donantes": len(users),
        "recurrentes": len(recurrentes),
        "una_vez": una_vez,
        "privadas": privadas,
        "con_mensaje": con_mensaje,
        "sin_mensaje": len(donations) - con_mensaje,
        "con_link": con_link,
        "promedio_por_donante": total / len(users) if users else 0.0,
        "donaciones_por_donante": len(donations) / len(users) if users else 0.0,
        "concentracion": {
            "top_1": share(max(1, len(ordered_values) // 100)),
            "top_10": share(max(1, len(ordered_values) // 10)),
            "top_10_donantes": share_from_users(users, 10, total),
        },
        "por_usuario_monto": sorted(users, key=lambda item: (-item["suma"], -item["cantidad"], item["nombre"]))[:top_n],
        "por_usuario_cantidad": sorted(users, key=lambda item: (-item["cantidad"], -item["suma"], item["nombre"]))[:top_n],
        "por_fecha": sorted(by_fecha.values(), key=lambda item: (-item["cantidad"], item["fecha"]))[:top_n],
        "por_periodo": [
            {"periodo": name, **by_bucket[name]}
            for name in _BUCKET_ORDEN
            if by_bucket.get(name, {}).get("cantidad")
        ],
        "por_rango": rangos,
        "mayores": sorted(donations, key=lambda item: (-donation_value(item), item["nombre"]))[:top_n],
    }


def share_from_users(users: list[dict], count: int, total: float) -> float:
    if not users or total <= 0:
        return 0.0
    top = sorted(users, key=lambda item: -item["suma"])[:count]
    return sum(item["suma"] for item in top) / total * 100


def _row(label: str, value: str) -> str:
    return f"  {label:<28} {value}"


def render_ranking(title: str, rows: list[tuple[str, str, str]]) -> list[str]:
    if not rows:
        return [title, "  (sin datos)", ""]
    name_w = max(len(row[0]) for row in rows)
    count_w = max(len(row[1]) for row in rows)
    lines = [title]
    for name, count, amount in rows:
        lines.append(f"  {name:<{name_w}}  {count:>{count_w}}  {amount}")
    lines.append("")
    return lines


def render_metricas(report: dict, user: str, filtros: str = "") -> str:
    extra = f"  filtro: {filtros}" if filtros else ""
    lines = [
        f"Análisis de donaciones — {user}",
        PAGE_URL if user.lower() == DEFAULT_USER else f"https://ceneka.net/{user}",
    ]
    if extra:
        lines.append(extra)
    lines.extend([
        "",
        "Totales",
        _row("Donaciones", f"{report['cantidad']:,}".replace(",", ".")),
        _row("Suma", format_amount(report["suma"])),
        _row("Promedio", format_amount(report["promedio"])),
        _row("Mediana", format_amount(report["mediana"])),
        _row("Mínimo", format_amount(report["minimo"])),
        _row("Máximo", format_amount(report["maximo"])),
        "",
        "Donantes",
        _row("Únicos", f"{report['donantes']:,}".replace(",", ".")),
        _row("Recurrentes (2+)", f"{report['recurrentes']:,}".replace(",", ".")),
        _row("Una sola vez", f"{report['una_vez']:,}".replace(",", ".")),
        _row("Promedio por donante", format_amount(report["promedio_por_donante"])),
        _row("Donaciones por donante", f"{report['donaciones_por_donante']:.2f}"),
        "",
        "Detalle",
        _row("Privadas", f"{report['privadas']:,}".replace(",", ".")),
        _row("Con mensaje", f"{report['con_mensaje']:,}".replace(",", ".")),
        _row("Sin mensaje", f"{report['sin_mensaje']:,}".replace(",", ".")),
        _row("Con link", f"{report['con_link']:,}".replace(",", ".")),
        "",
        "Concentración",
        _row("Top 1% donaciones", f"{report['concentracion']['top_1']:.1f}% del total"),
        _row("Top 10% donaciones", f"{report['concentracion']['top_10']:.1f}% del total"),
        _row("Top 10 donantes", f"{report['concentracion']['top_10_donantes']:.1f}% del total"),
        "",
    ])
    lines.extend(
        render_ranking(
            "Top donantes por monto",
            [
                (item["nombre"], f"{item['cantidad']} don.", format_amount(item["suma"]))
                for item in report["por_usuario_monto"]
            ],
        )
    )
    lines.extend(
        render_ranking(
            "Top donantes por cantidad",
            [
                (item["nombre"], f"{item['cantidad']} don.", format_amount(item["suma"]))
                for item in report["por_usuario_cantidad"]
            ],
        )
    )
    lines.extend(
        render_ranking(
            "Por período",
            [
                (item["periodo"], f"{item['cantidad']} don.", format_amount(item["suma"]))
                for item in report["por_periodo"]
            ],
        )
    )
    lines.extend(
        render_ranking(
            "Por fecha (texto de Ceneka)",
            [
                (item["fecha"], f"{item['cantidad']} don.", format_amount(item["suma"]))
                for item in report["por_fecha"]
            ],
        )
    )
    lines.extend(
        render_ranking(
            "Por rango de monto",
            [
                (item["etiqueta"], f"{item['cantidad']} don.", format_amount(item["suma"]))
                for item in report["por_rango"]
                if item["cantidad"]
            ],
        )
    )
    lines.extend(
        render_ranking(
            "Mayores donaciones",
            [
                (f"{item['nombre']} · {item['fecha']}", item["monto"], "")
                for item in report["mayores"]
            ],
        )
    )
    return "\n".join(line for line in lines if line is not None).rstrip() + "\n"


def render_tabla(donations: list[dict], user: str, todas: bool = False) -> str:
    titulo = f"Todas las donaciones ({len(donations)})" if todas else f"Donaciones ({len(donations)})"
    lines = [
        f"{titulo} — {user}",
        PAGE_URL if user.lower() == DEFAULT_USER else f"https://ceneka.net/{user}",
        "",
    ]
    if not donations:
        lines.append("No hay donaciones para mostrar.")
        return "\n".join(lines) + "\n"
    name_w = max(len(item["nombre"]) for item in donations)
    date_w = max((len(item["fecha"]) for item in donations), default=0)
    amount_w = max(len(item["monto"]) for item in donations)
    header = f"{'usuario':<{name_w}}  {'fecha':<{date_w}}  {'monto':>{amount_w}}"
    lines.append(header)
    lines.append("-" * len(header))
    for item in donations:
        lines.append(f"{item['nombre']:<{name_w}}  {item['fecha']:<{date_w}}  {item['monto']:>{amount_w}}")
    return "\n".join(lines) + "\n"


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


def describe_filters(args: argparse.Namespace) -> str:
    parts = []
    if args.donante:
        parts.append(f"usuario={args.donante}")
    if args.fecha:
        parts.append(f"fecha={args.fecha}")
    if args.minimo is not None:
        parts.append(f"min={args.minimo:g}")
    if args.maximo is not None:
        parts.append(f"max={args.maximo:g}")
    return ", ".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lista y analiza las donaciones públicas de una página de Ceneka."
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
        "--donante",
        help="filtra por nombre de quien donó (texto parcial, sin mayúsculas)",
    )
    parser.add_argument(
        "--fecha",
        help="filtra por fecha: hoy, semana, mes, año, o texto de Ceneka (ej. '7 horas')",
    )
    parser.add_argument(
        "--min",
        dest="minimo",
        type=float,
        help="monto mínimo (valor de Ceneka)",
    )
    parser.add_argument(
        "--max",
        dest="maximo",
        type=float,
        help="monto máximo (valor de Ceneka)",
    )
    parser.add_argument(
        "--metricas",
        action="store_true",
        help="mostrar el análisis (totales, donantes, fechas, rangos)",
    )
    parser.add_argument(
        "--tabla",
        action="store_true",
        help="listar usuario, fecha y monto en columnas",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=METRICAS_TOP,
        help=f"filas en los rankings (default: {METRICAS_TOP})",
    )
    parser.add_argument(
        "--desde",
        help="leer donaciones desde un JSON guardado en vez de Ceneka",
    )
    parser.add_argument(
        "--guardar",
        help="guardar el listado crudo en un JSON",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="imprimir JSON en lugar de texto",
    )
    args = parser.parse_args(argv)
    if args.cantidad is not None and args.cantidad < 1:
        parser.error("--cantidad tiene que ser 1 o más")
    if args.top < 1:
        parser.error("--top tiene que ser 1 o más")
    if args.minimo is not None and args.maximo is not None and args.minimo > args.maximo:
        parser.error("--min no puede ser mayor que --max")
    if not args.usuario.strip():
        parser.error("--usuario no puede estar vacío")
    args.usuario = args.usuario.strip()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    def on_progress(total: int) -> None:
        print(f"\rLeyendo donaciones… {total}", file=sys.stderr, end="", flush=True)

    try:
        if args.desde:
            donations = load_donations(args.desde)
        else:
            donations = fetch_latest(args.usuario, args.cantidad, on_progress=on_progress)
            print(file=sys.stderr)
    except FileNotFoundError:
        print(f"No está el archivo {args.desde}", file=sys.stderr)
        return 1
    except urllib.error.URLError as error:
        print(file=sys.stderr)
        print(f"No se pudo conectar con Ceneka: {error.reason}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, RuntimeError, TimeoutError, OSError) as error:
        if not args.desde:
            print(file=sys.stderr)
        print(f"No se pudieron leer las donaciones: {error}", file=sys.stderr)
        return 1

    if args.guardar:
        with open(args.guardar, "w", encoding="utf-8") as handle:
            json.dump(donations, handle, ensure_ascii=False)
            handle.write("\n")

    filtered = filter_donations(donations, args.donante, args.fecha, args.minimo, args.maximo)
    filtros = describe_filters(args)
    use_metricas = args.metricas or (
        not args.json and not args.tabla and args.cantidad is None
    )

    if args.json:
        payload = analyze(filtered, top=args.top) if args.metricas else filtered
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
        sys.stdout.write("\n")
    elif use_metricas:
        sys.stdout.write(render_metricas(analyze(filtered, top=args.top), args.usuario, filtros))
    elif args.tabla:
        sys.stdout.write(render_tabla(filtered, args.usuario, todas=args.cantidad is None))
    else:
        sys.stdout.write(render_text(filtered, args.usuario, todas=args.cantidad is None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
