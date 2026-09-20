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

Clic derecho → **Abrir**. No está notarizado por Apple.

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
| ONNX UVR | `mdx_models/*.onnx` (no se suben; van en el zip full) |
| RVC-WebUI (solo para **Entrenar**) | `git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI third_party/RVC-WebUI` |

```bash
python desktop.py
```

Tests:

```bash
.venv/bin/python -m unittest discover -s tests -q
```

No abras `desktop.py` y `./convert_rvc.sh` a la vez (OpenMP en Intel).

## Publicar este repo

El `main` local es la historia real. GitHub puede tener otra: **no hagas pull**. Con sesión de `gh`:

```bash
cd /Users/lucastomasi/grok/Audio_separator
gh auth login
git push -u --force-with-lease origin main
```

Eso pisa `origin/main`. No sube `dist/`, `library.json`, voces, ONNX ni RVC-WebUI.

## Licencia

MIT del glue: `LICENSE`. Modelos y webui de terceros: `NOTICE.md`.
