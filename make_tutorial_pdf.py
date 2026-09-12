#!/usr/bin/env python3
"""Build the Spanish user tutorial PDF for Audio Separator."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path("/Users/lucastomasi/grok/Audio_separator/Tutorial_Audio_Separator.pdf")
DESKTOP = Path("/Users/lucastomasi/Desktop/Tutorial_Audio_Separator.pdf")

VIOLET = colors.HexColor("#6D28D9")
VIOLET_SOFT = colors.HexColor("#EDE9FE")
PINK = colors.HexColor("#DB2777")
INK = colors.HexColor("#18181B")
MUTED = colors.HexColor("#52525B")
LINE = colors.HexColor("#D4D4D8")
PAPER = colors.HexColor("#FAFAF9")
WHITE = colors.white


def styles():
    base = getSampleStyleSheet()
    return {
        "kicker": ParagraphStyle(
            "kicker",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            textColor=VIOLET,
            spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "lede": ParagraphStyle(
            "lede",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15.5,
            textColor=MUTED,
            alignment=TA_JUSTIFY,
            spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=VIOLET,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "cell": ParagraphStyle(
            "cell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.4,
            textColor=INK,
        ),
        "cellb": ParagraphStyle(
            "cellb",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12.4,
            textColor=INK,
        ),
        "headcell": ParagraphStyle(
            "headcell",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=WHITE,
        ),
        "stepn": ParagraphStyle(
            "stepn",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            textColor=VIOLET,
            alignment=TA_CENTER,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            textColor=MUTED,
            spaceBefore=2,
            spaceAfter=10,
        ),
        "tip": ParagraphStyle(
            "tip",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=INK,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "li": ParagraphStyle(
            "li",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13.6,
            textColor=INK,
        ),
    }


def P(text, style):
    return Paragraph(text, style)


def numbered_steps(s):
    rows = [
        [
            P("1", s["stepn"]),
            P("<b>Abre la app</b><br/>Doble clic en <b>Audio Separator</b> del Escritorio. El primer arranque puede tardar 1 o 2 minutos. El navegador se abre solo en http://127.0.0.1:7860", s["cell"]),
        ],
        [
            P("2", s["stepn"]),
            P("<b>Elige el audio</b><br/>Sube un archivo (mp3, wav, m4a, etc.) o pega un enlace de YouTube y pulsa <b>Descargar audio</b>. Puedes escucharlo en el reproductor antes de continuar.", s["cell"]),
        ],
        [
            P("3", s["stepn"]),
            P("<b>Elige qué extraer</b><br/>Marca <b>Voz</b>, <b>Instrumental</b> o las dos. Opcional: solo voz principal, quitar reverb, o efectos.", s["cell"]),
        ],
        [
            P("4", s["stepn"]),
            P("<b>Separa y descarga</b><br/>Elige formato (WAV, MP3 o FLAC) y pulsa <b>Separar audio</b>. Cuando termine, baja los archivos desde <b>Archivos listos</b>.", s["cell"]),
        ],
        [
            P("5", s["stepn"]),
            P("<b>Cierra la app</b><br/>Pulsa <b>Cerrar app</b> en el cuadro de diálogo. Si la abriste por Terminal, cierra esa ventana. No queda corriendo en segundo plano.", s["cell"]),
        ],
    ]
    table = Table(rows, colWidths=[0.55 * inch, 6.15 * inch])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (0, -1), VIOLET_SOFT),
                ("BACKGROUND", (1, 0), (1, -1), PAPER),
                ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                ("LINEBELOW", (0, 0), (-1, -2), 0.3, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def pipeline(s):
    cells = [
        P("<b>1. Entrada</b><br/>Archivo local o audio bajado de YouTube (mp3).", s["cell"]),
        P("<b>2. Preparar</b><br/>Se convierte a estéreo WAV a 44.1 kHz.", s["cell"]),
        P("<b>3. Modelo</b><br/>MDX-Net (ONNX) estima voz vs. resto en CPU.", s["cell"]),
        P("<b>4. Salida</b><br/>Pistas por separado, con efectos si los pediste.", s["cell"]),
    ]
    grid = Table(
        [[cells[0], cells[1]], [cells[2], cells[3]]],
        colWidths=[3.35 * inch, 3.35 * inch],
    )
    grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, -1), VIOLET_SOFT),
                ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return grid


def options_table(s):
    data = [
        [
            P("Opción", s["headcell"]),
            P("Qué hace", s["headcell"]),
        ],
        [
            P("<b>Voz</b>", s["cellb"]),
            P("Aisla canto o voz. Usa el modelo UVR-MDX-NET-Voc_FT.", s["cell"]),
        ],
        [
            P("<b>Instrumental</b>", s["cellb"]),
            P("Saca el fondo (música sin voz) con UVR-MDX-NET-Inst_HQ_4.", s["cell"]),
        ],
        [
            P("<b>Solo voz principal</b>", s["cellb"]),
            P("Intenta dejar atrás coros y doblajes (modelo KARA). Tarda más.", s["cell"]),
        ],
        [
            P("<b>Quitar reverb</b>", s["cellb"]),
            P("Reduce eco de sala en la voz (Reverb HQ). Tarda más.", s["cell"]),
        ],
        [
            P("<b>Efectos</b>", s["cellb"]),
            P("Reverb, delay, compresor y ganancia sobre la pista ya separada. No mejoran la separación; solo el color del sonido.", s["cell"]),
        ],
        [
            P("<b>Formato</b>", s["cellb"]),
            P("WAV (sin pérdida, más pesado), MP3 (más liviano) o FLAC (sin pérdida comprimido).", s["cell"]),
        ],
    ]
    table = Table(data, colWidths=[1.7 * inch, 5.0 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), VIOLET),
                ("BACKGROUND", (0, 1), (-1, 1), PAPER),
                ("BACKGROUND", (0, 3), (-1, 3), PAPER),
                ("BACKGROUND", (0, 5), (-1, 5), PAPER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def tip_box(s, title, body):
    inner = Table(
        [[P(f"<b>{title}</b><br/>{body}", s["tip"])]],
        colWidths=[6.7 * inch],
    )
    inner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), VIOLET_SOFT),
                ("BOX", (0, 0), (-1, -1), 0.6, VIOLET),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return inner


def draw_page(canvas, doc):
    canvas.saveState()
    width, height = letter
    canvas.setFillColor(VIOLET)
    canvas.rect(0, height - 28, width, 28, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(0.8 * inch, height - 18, "AUDIO SEPARATOR")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(width - 0.8 * inch, height - 18, "Tutorial de uso")

    canvas.setFillColor(PINK)
    canvas.rect(0, 0, width, 32, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.8 * inch, 13, "Uso local en este Mac  ·  No necesita Grok CLI")
    canvas.drawRightString(width - 0.8 * inch, 13, f"{doc.page}")
    canvas.restoreState()


def build():
    s = styles()
    story = []

    story.append(P("ESTUDIO LOCAL", s["kicker"]))
    story.append(P("Cómo funciona Audio Separator", s["h1"]))
    story.append(
        P(
            "Audio Separator es una app en tu Mac que parte una canción en dos: "
            "la <b>voz</b> y el <b>instrumental</b>. Corre solo en este equipo, "
            "sin subir el audio a internet (salvo que pegues un enlace de YouTube). "
            "No hace falta dejarla abierta: la abres, la usas y la cierras.",
            s["lede"],
        )
    )

    story.append(P("Cómo se usa", s["h2"]))
    story.append(numbered_steps(s))
    story.append(
        P(
            "El icono está en el Escritorio y en Aplicaciones. También puedes abrir "
            "<b>Audio Separator.command</b> dentro de la carpeta del proyecto si prefieres ver la terminal.",
            s["caption"],
        )
    )

    story.append(P("Qué ocurre por dentro", s["h2"]))
    story.append(
        P(
            "No &quot;borra&quot; la voz a mano. Un modelo de redes neuronales (MDX-Net, el mismo "
            "enfoque de Ultimate Vocal Remover) escucha el espectro de la canción y estima "
            "qué parte es canto y qué parte es el resto. Esa estimación corre con ONNX Runtime "
            "en la CPU de este Mac Intel: por eso un tema de 3 minutos puede tardar varios minutos.",
            s["body"],
        )
    )
    story.append(pipeline(s))
    story.append(
        P(
            "Los modelos ya están descargados en la carpeta mdx_models. YouTube usa yt-dlp "
            "con Node para resolver las firmas del video, y FFmpeg para dejar el audio en mp3.",
            s["caption"],
        )
    )

    story.append(
        KeepTogether(
            [
                P("Opciones de la pantalla", s["h2"]),
                options_table(s),
                Spacer(1, 10),
            ]
        )
    )

    story.append(P("YouTube", s["h2"]))
    story.append(
        P(
            "Pega el enlace (youtube.com, youtu.be o sin https) y pulsa <b>Descargar audio</b>. "
            "Cuando termine, el reproductor muestra el tema y ya puedes separar. "
            "La primera vez puede tardar un poco; no cierres la ventana mientras descarga. "
            "Sirve un video suelto, no una lista de reproducción completa.",
            s["body"],
        )
    )

    story.append(P("Si algo falla", s["h2"]))
    bullets = ListFlowable(
        [
            ListItem(P("La app no abre el navegador: espera 1-2 minutos el primer arranque y entra a http://127.0.0.1:7860", s["li"])),
            ListItem(P("YouTube no descarga: revisa que el enlace sea de un video público y vuelve a pulsar Descargar audio.", s["li"])),
            ListItem(P("Tarda mucho: es normal en este Mac (sin GPU). Prueba primero un clip corto o el ejemplo de la app.", s["li"])),
            ListItem(P("El resultado suena sucio: prueba sin efectos, o marca Quitar reverb / Solo voz principal. No hay separación perfecta.", s["li"])),
            ListItem(P("Quedó colgada: Cerrar app o cierra la Terminal. Vuelve a abrirla; no hace falta Grok CLI.", s["li"])),
        ],
        bulletType="bullet",
        leftIndent=14,
        bulletFontName="Helvetica",
        bulletFontSize=9,
        spaceBefore=0,
        spaceAfter=8,
    )
    story.append(bullets)

    story.append(
        tip_box(
            s,
            "Consejo",
            "Empieza con un fragmento corto y formato MP3. Si te gusta el resultado, "
            "repite con el tema completo en WAV o FLAC. Deja la ventana de la app abierta "
            "mientras procesa; al terminar, cierra con Cerrar app.",
        )
    )

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.65 * inch,
        title="Cómo funciona Audio Separator",
        author="Audio Separator",
        subject="Tutorial de uso local",
    )
    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    DESKTOP.write_bytes(OUT.read_bytes())
    print(OUT)
    print(DESKTOP)


if __name__ == "__main__":
    build()
