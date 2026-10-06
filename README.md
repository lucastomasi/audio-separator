# Audio Separator

App local para Mac **Intel** (o Apple Silicon con Rosetta): separar voz/instrumental, entrenar y convertir voz con RVC, unir pistas. El audio queda en disco.

YouTube, Edge TTS y Completar instalación (Hugging Face / pesos) usan red. Separar, Entrenar y Convertir van offline solo si los modelos ya están en disco.

## Requisitos

- macOS 13+
- CPU x86_64 o Rosetta 2
- Python 3.12 para desarrollo
- ~3 GB libres para el `.app` standalone

## App empaquetada

| Zip | Qué trae |
|---|---|
| `Audio-Separator-macOS-Intel.zip` | Full, pesos adentro (~2 GB) |
| `Audio-Separator-macOS-Intel-Lite.zip` | Sin ~700 MB de RVC; **Completar instalación** la primera vez |

**Entregable:** `dist/Audio-Separator-macOS-Intel.zip`. Copiá `Audio Separator.app` a Aplicaciones. Clic derecho → **Abrir**. No hace falta Terminal ni Grok. No está notarizado por Apple.

El alias de desarrollo en `~/grok` **no** es el producto.

Los zip **no** van en git. Se arman con:

```bash
./scripts/build_standalone.sh
./scripts/build_standalone_lite.sh
```

## Flujo

1. **Canción** — archivo o YouTube  
2. **Extraer** — voz / instrumental (minutos en Intel)  
3. **Resultado** — `~/Downloads/Audio Separator`  
4. **Voz (RVC)** — Entrenar (cada epoch se guarda) → Convertir  
5. **Unir**  
6. **Texto → habla** — Edge (internet) → tu `.pth` RVC  

Cerrar la ventana no corta un train ya largado. No relances el mismo nombre si sigue corriendo.

## Desarrollo

```bash
git clone https://github.com/lucastomasi/audio-separator.git
cd audio-separator
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-macos.txt
```

### Lo que no viene en el clone

| Hace falta | Dónde |
|---|---|
| Pesos RVC (HuBERT, RMVPE, f0G/D) | App → **Completar instalación**, o `library/models/rvc/` |
| ONNX UVR | `mdx_models/*.onnx` (no se suben). El zip **full** es un enlatado: los copia del Mac de build. Sin esos archivos `build_standalone.sh` aborta. |
| RVC-WebUI (solo para **Entrenar**) | `git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI third_party/RVC-WebUI` |

```bash
python desktop.py
```

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
