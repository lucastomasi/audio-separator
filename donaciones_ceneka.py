#!/usr/bin/env python3
"""Muestra y analiza las donaciones de https://ceneka.net/losherederosdealberdi.

Usa solo la biblioteca estándar, así corre en la terminal de Mac con python3.
Sin -n recorre todas las páginas (el listado público son unas decenas de miles):

    python3 donaciones_ceneka.py --metricas
    python3 donaciones_ceneka.py --ciencia --desde donaciones.json
    python3 donaciones_ceneka.py --metricas --donante Disociandri --min 500
    python3 donaciones_ceneka.py --tabla --fecha hoy
    python3 donaciones_ceneka.py -n 20
    python3 donaciones_ceneka.py --json --guardar donaciones.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import statistics
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
_FREQ_DONANTE = (
    (1, 1, "1 donación"),
    (2, 4, "2 – 4"),
    (5, 19, "5 – 19"),
    (20, 99, "20 – 99"),
    (100, None, "100 o más"),
)
_PERCENTILES = (10, 25, 50, 75, 90, 95, 99)


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
    return percentile(values, 50)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def gini_coefficient(values: list[float]) -> float:
    ordered = sorted(v for v in values if v >= 0)
    if not ordered:
        return 0.0
    total = sum(ordered)
    if total == 0:
        return 0.0
    n = len(ordered)
    weighted = sum(index * value for index, value in enumerate(ordered, start=1))
    return (2 * weighted) / (n * total) - (n + 1) / n


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def moment_skewness(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    mean = sum(values) / len(values)
    second = sum((value - mean) ** 2 for value in values) / len(values)
    third = sum((value - mean) ** 3 for value in values) / len(values)
    if second == 0:
        return 0.0
    return third / (second ** 1.5)


def pareto_cutoff(values: list[float], target: float = 0.8) -> dict:
    if not values:
        return {"items": 0, "porcentaje_items": 0.0, "alcanzado": 0.0}
    ordered = sorted(values, reverse=True)
    total = sum(ordered)
    if total <= 0:
        return {"items": 0, "porcentaje_items": 0.0, "alcanzado": 0.0}
    acc = 0.0
    for index, value in enumerate(ordered, start=1):
        acc += value
        if acc / total >= target:
            return {
                "items": index,
                "porcentaje_items": index / len(ordered) * 100,
                "alcanzado": acc / total * 100,
            }
    return {"items": len(ordered), "porcentaje_items": 100.0, "alcanzado": 100.0}


def ascii_bar(value: float, peak: float, width: int = 28) -> str:
    if peak <= 0 or value <= 0:
        return ""
    return "█" * max(1, round(value / peak * width))


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
    page = 0
    while count is None or len(collected) < count:
        if page >= MAX_PAGES:
            break
        chunk = fetcher(user, page, page_size)
        if not chunk:
            break
        for item in chunk:
            collected.append(normalize_donation(item))
            if count is not None and len(collected) >= count:
                break
        if on_progress is not None:
            on_progress(len(collected))
        if len(chunk) < page_size:
            break
        page += 1
    return collected


def load_donations(path: str) -> list[dict]:
    from pathlib import Path

    target = Path(path)
    opener = gzip.open if target.suffix == ".gz" else target.open
    with opener(target, "rt", encoding="utf-8") as handle:
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
    q1 = percentile(values, 25)
    q3 = percentile(values, 75)
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr
    outliers = [value for value in values if value > fence]
    user_sums = [item["suma"] for item in users]
    user_counts = [item["cantidad"] for item in users]
    msg_lens = [len(item.get("mensaje") or "") for item in donations]
    freq = []
    for low, high, label in _FREQ_DONANTE:
        selected = [
            item for item in users
            if item["cantidad"] >= low and (high is None or item["cantidad"] <= high)
        ]
        freq.append({
            "etiqueta": label,
            "donantes": len(selected),
            "donaciones": sum(item["cantidad"] for item in selected),
            "suma": sum(item["suma"] for item in selected),
        })
    return {
        "cantidad": len(donations),
        "suma": total,
        "promedio": total / len(values) if values else 0.0,
        "mediana": median(values),
        "minimo": min(values) if values else 0.0,
        "maximo": max(values) if values else 0.0,
        "desvio": statistics.pstdev(values) if len(values) > 1 else 0.0,
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
            "gini_donaciones": gini_coefficient(values),
            "gini_donantes": gini_coefficient(user_sums),
            "pareto_80_donaciones": pareto_cutoff(values, 0.8),
            "pareto_80_donantes": pareto_cutoff(user_sums, 0.8),
        },
        "ciencia": {
            "percentiles": {f"p{p}": percentile(values, p) for p in _PERCENTILES},
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "asimetria": moment_skewness(values),
            "outliers_iqr": len(outliers),
            "outliers_suma": sum(outliers),
            "cerca_monto_mensaje": pearson(values, [float(length) for length in msg_lens]),
            "cerca_frecuencia_monto": pearson(
                [float(count) for count in user_counts],
                user_sums,
            ),
            "frecuencia_donante": freq,
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


def _fmt_n(number: float) -> str:
    if float(number).is_integer():
        return f"{int(number):,}".replace(",", ".")
    return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pareto_line(cut: dict, unidad: str) -> str:
    items = _fmt_n(cut.get("items") or 0)
    pct = cut.get("porcentaje_items") or 0
    return f"{items} {unidad} ({pct:.1f}%) concentran el 80%"


def render_ciencia(report: dict) -> list[str]:
    science = report.get("ciencia") or {}
    if not science:
        return []
    percentiles = science.get("percentiles") or {}
    lines = [
        "Data science",
        _row("P10 / P25 / P50", "  ".join(format_amount(percentiles.get(key, 0)) for key in ("p10", "p25", "p50"))),
        _row("P75 / P90 / P99", "  ".join(format_amount(percentiles.get(key, 0)) for key in ("p75", "p90", "p99"))),
        _row("IQR (P75 − P25)", format_amount(science.get("iqr", 0))),
        _row("Asimetría", f"{science.get('asimetria', 0):.2f}"),
        _row("Outliers (IQR × 1.5)", f"{_fmt_n(science.get('outliers_iqr', 0))} · {format_amount(science.get('outliers_suma', 0))}"),
        _row("r monto ↔ mensaje", f"{science.get('cerca_monto_mensaje', 0):.3f}"),
        _row("r freq ↔ monto donante", f"{science.get('cerca_frecuencia_monto', 0):.3f}"),
        "",
    ]
    lines.extend(render_insights(report))
    freq = science.get("frecuencia_donante") or []
    peak = max((item["donantes"] for item in freq), default=0)
    lines.extend(
        render_ranking(
            "Frecuencia por donante",
            [
                (
                    f"{item['etiqueta']}  {ascii_bar(item['donantes'], peak)}",
                    f"{item['donantes']} pers.",
                    format_amount(item["suma"]),
                )
                for item in freq
                if item["donantes"]
            ],
        )
    )
    peak_rango = max((item["cantidad"] for item in report.get("por_rango") or []), default=0)
    lines.extend(
        render_ranking(
            "Histograma de montos",
            [
                (
                    f"{item['etiqueta']}  {ascii_bar(item['cantidad'], peak_rango)}",
                    f"{item['cantidad']} don.",
                    format_amount(item["suma"]),
                )
                for item in report.get("por_rango") or []
                if item["cantidad"]
            ],
        )
    )
    return lines


def insights_list(report: dict) -> list[str]:
    science = report.get("ciencia") or {}
    conc = report.get("concentracion") or {}
    notes = []
    mean = report.get("promedio") or 0
    med = report.get("mediana") or 0
    if mean > med * 1.5:
        notes.append(f"La media ({format_amount(mean)}) está muy por encima de la mediana ({format_amount(med)}): pocos montos altos tiran el promedio.")
    gini_users = conc.get("gini_donantes") or 0
    if gini_users >= 0.7:
        notes.append(f"Gini de donantes {gini_users:.3f}: el dinero está muy concentrado en un grupo chico.")
    elif gini_users >= 0.4:
        notes.append(f"Gini de donantes {gini_users:.3f}: hay desigualdad, pero no extrema.")
    pareto = conc.get("pareto_80_donantes") or {}
    if pareto.get("porcentaje_items"):
        notes.append(f"El 80% del monto lo aportan {_fmt_n(pareto['items'])} donantes ({pareto['porcentaje_items']:.1f}% del grupo).")
    outliers = science.get("outliers_iqr") or 0
    if outliers:
        share = (science.get("outliers_suma") or 0) / report["suma"] * 100 if report.get("suma") else 0
        notes.append(f"Hay {_fmt_n(outliers)} outliers de monto; suman {format_amount(science.get('outliers_suma', 0))} ({share:.1f}% del total).")
    r_freq = science.get("cerca_frecuencia_monto") or 0
    if r_freq >= 0.4:
        notes.append(f"Quienes donan más veces también aportan más plata (r={r_freq:.2f}).")
    elif abs(r_freq) < 0.2:
        notes.append(f"Donar muchas veces no implica aportar más plata (r={r_freq:.2f}).")
    if report.get("recurrentes") and report.get("donantes"):
        pct = report["recurrentes"] / report["donantes"] * 100
        notes.append(f"El {pct:.1f}% de los donantes volvió al menos una vez.")
    return notes


def render_insights(report: dict) -> list[str]:
    lines = ["Lectura"]
    lines.extend(f"  {note}" for note in insights_list(report))
    lines.append("")
    return lines


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
        _row("Desvío", format_amount(report.get("desvio", 0))),
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
        _row("Gini (donaciones)", f"{report['concentracion']['gini_donaciones']:.3f}"),
        _row("Gini (donantes)", f"{report['concentracion']['gini_donantes']:.3f}"),
        _row("Pareto 80% monto", _pareto_line(report["concentracion"]["pareto_80_donaciones"], "donaciones")),
        _row("Pareto 80% donantes", _pareto_line(report["concentracion"]["pareto_80_donantes"], "donantes")),
        "",
    ])
    lines.extend(render_ciencia(report))
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
        "--ciencia",
        action="store_true",
        help="igual que --metricas: análisis estadístico del grupo",
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
    use_metricas = args.metricas or args.ciencia or (
        not args.json and not args.tabla and args.cantidad is None
    )

    if args.json:
        payload = analyze(filtered, top=args.top) if (args.metricas or args.ciencia) else filtered
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
