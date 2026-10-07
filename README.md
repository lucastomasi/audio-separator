# Audio Separator

App **local** para Mac **Intel** (o Apple Silicon con Rosetta): separar voz/instrumental, entrenar y convertir voz con RVC, unir pistas. Todo el audio se procesa en esta máquina; el resultado queda en disco. **No es un servicio online.**

Con el zip full enlatado, Separar / Entrenar / Convertir / Unir van **sin internet**. YouTube, Edge TTS y Completar instalación (solo el zip lite o un clone) son opcionales y sí usan red.

## Requisitos

- macOS 13+
- CPU x86_64 o Rosetta 2
- Python 3.12 (python.org o Homebrew) — lo usa el `.app` del clone la primera vez
- ffmpeg: el zip full ya lo copia. En un clone, `requirements-macos.txt` instala `imageio-ffmpeg` (no hace falta `brew`) si no hay un ffmpeg en el PATH
- ~3 GB libres para el `.app` standalone

## Arranque con un clic (sin Terminal)

El repo trae `Audio Separator.app` al lado de `desktop.py`. En el Finder: clic derecho → **Abrir**.

Ese `.app` hace todo solo: crea `.venv` y `.venv-vc` si faltan, instala `requirements-macos.txt` + `requirements-vc.txt`, arranca el worker de conversión y abre la ventana nativa. El primer clic baja PyTorch 2.2.2 y puede tardar varios minutos. El log queda en `~/Library/Logs/Audio Separator/launch.log`.

Para regenerar el `.app` (no hace falta en un clone normal):

```bash
./scripts/build_launcher_app.sh
```

## App empaquetada (zip full / lite)

| Zip | Qué trae |
|---|---|
| `Audio-Separator-macOS-Intel.zip` | Full, pesos adentro (~2 GB) |
| `Audio-Separator-macOS-Intel-Lite.zip` | Sin ~700 MB de RVC; **Completar instalación** la primera vez |

**Entregable:** `dist/Audio-Separator-macOS-Intel.zip`. Copiá `Audio Separator.app` a Aplicaciones. Clic derecho → **Abrir**. No hace falta Terminal ni Grok. No está notarizado por Apple.

El alias de desarrollo en `~/grok` **no** es el producto.

Los zip **no** van en git. Se arman en un Mac Intel (o Rosetta) que ya tenga Python 3.12, ffmpeg y los pesos:

```bash
./scripts/build_standalone.sh        # si faltan .venv / .venv-vc, los crea
./scripts/build_standalone_lite.sh
```

`build_standalone.sh` copia los venvs al `.app`. `macos_launcher.sh` (el ejecutable del `.app`) vuelve a instalar deps si el zip llegó incompleto y el Mac tiene Python 3.12.

## Flujo

1. **Canción** — archivo local (YouTube es opcional y usa red)  
2. **Separar** — voz / instrumental en este Mac  
3. **Resultado** — `~/Downloads/Audio Separator`  
4. **Voz (RVC)** — Entrenar (cada epoch se guarda) → Convertir  
5. **Unir**  
6. **Texto → habla** — opcional: Edge/ElevenLabs (internet) → tu `.pth` RVC  

Cerrar la ventana no corta un train ya largado. No relances el mismo nombre si sigue corriendo.

## Desarrollo (manual, equivalente al .app)

```bash
git clone https://github.com/lucastomasi/audio-separator.git
cd audio-separator
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-macos.txt
python3.12 -m venv .venv-vc
.venv-vc/bin/pip install -r requirements-vc.txt
python desktop.py
```

O: `bash scripts/macos_launcher.sh` (mismo bootstrap que el `.app`).

Hay **dos venvs a propósito**. Un solo environment no puede satisfacer Gradio 6.20 y el stack Applio a la vez:

| Archivo | Para | Pins que importan |
|---|---|---|
| `requirements-macos.txt` | UI / UVR / desktop | `torch==2.2.2`, `gradio==6.20.0`, `huggingface-hub>=1.2,<2`, `imageio-ffmpeg`. Sin transformers, coqui-tts ni infer-rvc-python. |
| `requirements-vc.txt` | Convertir + Entrenar (`third_party/vc`) | `torch==2.2.2`, `transformers==4.53.3`, `huggingface-hub==0.36.2`, `librosa>=0.10,<0.11`, `faiss-cpu==1.10.0` (wheel x86_64 macOS 13), `imageio-ffmpeg` |
| `requirements.txt` | Space/Linux | `torch==2.9.1` (no lo uses en el Mac Intel) |

`torch==2.2.2` es el último wheel x86_64 de macOS. Coqui/XTTS no entra en el venv de la app; el TTS de la UI es Edge/ElevenLabs → RVC.

### Lo que no viene en el clone

| Hace falta | Dónde |
|---|---|
| Pesos RVC (HuBERT, RMVPE, f0G/D) | App → **Completar instalación**, o `library/models/rvc/` |
| ONNX UVR | `mdx_models/*.onnx` (no se suben). El zip **full** es un enlatado: los copia del Mac de build. Sin esos archivos `build_standalone.sh` aborta. |

## iPhone (Safari, agregar a inicio)

No es una app nativa ni un IPA. Es esta misma página en la pantalla de inicio.

`desktop.py` abre la ventana en la Mac. Para llegar desde el iPhone, en la computadora que procesa el audio:

```bash
AUDIO_SEPARATOR_HOST=0.0.0.0 python app.py
```

La terminal imprime una URL con `access_token`. Esa URL es la llave: no la pases. En el iPhone, en la misma red, abrila en Safari usando la IP de esa computadora (no `127.0.0.1`). Después: Compartir → **Agregar a inicio**. El ícono abre Audio Separator sin la barra de Safari.

Demucs y RVC, y la separación UVR, corren en esa computadora o en la GPU en la nube que hospeda el servidor. El teléfono solo muestra la interfaz. Sin `AUDIO_SEPARATOR_HOST`, el servidor sigue cerrado a localhost. Si el ícono abre un error, volvé a abrir la URL impresa en Safari y agregalo de nuevo.



Tests:

```bash
.venv/bin/python -m unittest discover -s tests -q
```

No abras `desktop.py` y `./convert_rvc.sh` a la vez (OpenMP en Intel).

## Licencia

MIT del glue: `LICENSE`. Modelos y webui de terceros: `NOTICE.md`.
