# Audio Separator

App local para Mac: separar voz e instrumental, bajar audio de YouTube, convertir voz con RVC en disco y volver a unir.

No usa el Space de Hugging Face.

## Instalar en Mac

Para Apple Silicon, bajá el disco (trae Python y las librerías; no hace falta instalar nada más):

https://github.com/lucastomasi/audio-separator/releases/download/macos-arm64/Audio-Separator-arm64.dmg

Abrilo. La app está a la izquierda y Aplicaciones a la derecha. Arrastrá **Audio Separator** a **Aplicaciones**, cerrá el disco y no la abras desde adentro. `LEEME.txt` está en el mismo disco.

En Aplicaciones, **Control-clic** (clic derecho) sobre Audio Separator → **Abrir** → **Abrir**.

macOS puede decir que no puede verificar al desarrollador. La app no está firmada. Ese Control-clic → Abrir es el paso normal.

Si igual no abre, en Terminal:

```bash
xattr -dr com.apple.quarantine "/Applications/Audio Separator.app"
```

Volvé a abrirla con Control-clic → Abrir. La primera ventana puede tardar uno o dos minutos. Si falla, el detalle queda en `~/Library/Logs/Audio Separator/launch.log`.

### Modelos y procesador

La separación local (canal central) funciona sin bajar nada. Es más simple que un modelo de separación.

Estos archivos son grandes y no van en el DMG. La app no los descarga:

- Un `.onnx` para separar mejor, en `~/Library/Application Support/Audio Separator/mdx_models/`.
- Para convertir la voz, en `~/Library/Application Support/Audio Separator/rvc_models/`:
  - `hubert_base/` — carpeta con `config.json` y los pesos (`model.safetensors` o `pytorch_model.bin`). Un `hubert_base.pt` suelto no alcanza.
  - `rmvpe.pt`
  - `tu-voz.pth`
  - `tu-voz.index` — opcional.

No hace falta una GPU NVIDIA. La separación ONNX corre en CPU. La conversión de voz puede usar el chip de Apple si PyTorch lo detecta; si no, usa CPU y tarda más.

YouTube usa el ffmpeg que viene dentro del `.app`. Si una descarga falla, instalá ffmpeg en la Mac (`brew install ffmpeg`) y volvé a abrir.

### Armar el .app y el DMG

En una Mac, desde el repo:

```bash
bash macos/build_release.sh
```

Eso deja `dist/Audio Separator.app` y `dist/Audio-Separator-arm64.dmg`. El script embebe un CPython 3.12, compila `macos/launcher.c` como ejecutable Mach-O y arma el disco con la app a la izquierda y Aplicaciones a la derecha.

El ejecutable no es un script. `clang` lo genera en la Mac del armado y le pone una firma ad-hoc, sin certificado de desarrollador y sin notarización. Gatekeeper igual pide Control-clic → Abrir.

En una Mac Intel, el mismo comando genera el disco para esa máquina. El enlace de arriba es solo Apple Silicon.

`bash macos/build_app.sh --layout-only` crea la carpeta del `.app` sin Python y sin el binario.

## Desarrollo

```bash
.venv/bin/python desktop.py
```

## Modelos RVC

Copiá estos archivos a `rvc_models/` antes de convertir. La app no los descarga. Si la corrés desde el repo, esa carpeta está al lado del código. Si abrís el `.app`, es `~/Library/Application Support/Audio Separator/rvc_models/`.

- `hubert_base/` — carpeta del modelo HuBERT, con `config.json` y los pesos (`model.safetensors` o `pytorch_model.bin`). Un `hubert_base.pt` suelto no alcanza.
- `rmvpe.pt` — el estimador de pitch.
- `tu-voz.pth` — el modelo de voz.
- `tu-voz.index` — opcional, el índice de esa voz.

Esos binarios quedan fuera de git. En la app: Actualizá los modelos y después Convertí la voz. El pitch es rmvpe; no hace falta pyworld.

`infer-rvc-python` pide `pyworld==0.3.4`, que no tiene wheel para Python 3.12. Si `pip install -r requirements-macos.txt` se cae ahí, instalá ese paquete con `--no-deps` después del resto. La app reemplaza pyworld por un stub y no lo llama.
