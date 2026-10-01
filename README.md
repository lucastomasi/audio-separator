# Audio Separator

App local para Mac: separar voz e instrumental, bajar audio de YouTube, convertir voz con RVC en disco y volver a unir.

No usa el Space de Hugging Face.

## Instalar en Mac

La descarga es un disco `Audio Separator.dmg`. Adentro está `Audio Separator.app`, con Python y las librerías ya instalados. No hace falta instalar Python ni armar un entorno a mano.

El DMG lo arma el workflow de GitHub Actions **macOS app** (corre en cada pull request y también a mano). Entrá al workflow, abrí el run verde y bajá el artifact `Audio-Separator-macos-arm64`. Ese disco es para Mac con Apple Silicon. En una Mac Intel, el mismo script (`bash macos/build_release.sh`) genera el disco para esa máquina.

1. Abrí el DMG.
2. Arrastrá **Audio Separator** a la carpeta **Aplicaciones**. Leé `LEEME.txt` si está al lado.
3. Cerrá el DMG. No abras la app desde adentro del disco.
4. En Aplicaciones, **Control-clic** (clic derecho) sobre Audio Separator → **Abrir** → **Abrir**.

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

Eso deja `dist/Audio Separator.app` y `dist/Audio Separator.dmg`. El script baja un CPython 3.12 relocatable, instala las dependencias dentro del bundle, compila `macos/launcher.c` como ejecutable Mach-O de la app y arma el disco con `hdiutil`.

El ejecutable no es un script. Un `.app` cuyo programa principal es un shell, macOS lo trata como dañado y la ventana no queda como la app que abriste. `clang` genera el binario en la Mac que arma el disco. Esa misma Mac le pone una firma ad-hoc (`codesign -s -`) al ejecutable y a ffmpeg, sin certificado de desarrollador y sin notarización. Gatekeeper igual pide Control-clic → Abrir.

El artifact del workflow es un zip. Adentro está `Audio Separator.dmg`.

`bash macos/build_app.sh --layout-only` solo crea la carpeta del `.app`, sin Python y sin el binario. Sirve para revisar la estructura en cualquier sistema.

## Desarrollo

```bash
.venv/bin/python desktop.py
```

Eso abre la ventana local. No se publica a la red.

## iPhone (pantalla de inicio, no es la App Store)

No es una app nativa ni un IPA. Es esta misma página de Gradio, guardada en el inicio del iPhone. Safari la abre sin la barra del navegador. El teléfono no procesa el audio.

En la computadora, desde la carpeta del proyecto:

```bash
.venv/bin/python app.py
```

Eso escucha en el puerto 7860 de la red local. En la terminal aparece una dirección `http://192.168.x.x:7860`. El iPhone tiene que estar en la misma Wi-Fi. Si el sistema pregunta, permití las conexiones entrantes.

1. Abrí esa dirección en Safari.
2. Tocá Compartir.
3. Tocá Agregar a pantalla de inicio y confirmá.

El ícono queda al lado de las otras apps. Adentro sigue siendo el servidor de la computadora.

RVC y los modelos `.onnx` de `mdx_models` corren en esa computadora, que es donde están los archivos. Si la apagás, o si el teléfono sale de la Wi-Fi, el ícono no puede separar ni convertir nada. No hace falta una cuenta de desarrollador ni subir nada a la App Store.

Subí el audio como archivo. En una dirección `http://` de la red, Safari suele bloquear el micrófono del iPhone.

Otro puerto: `.venv/bin/python app.py --port 7861`. Solo en esta máquina, sin el teléfono: `.venv/bin/python app.py --host 127.0.0.1`.

## Modelos RVC

Copiá estos archivos a `rvc_models/` antes de convertir. La app no los descarga. Si la corrés desde el repo, esa carpeta está al lado del código. Si abrís el `.app`, es `~/Library/Application Support/Audio Separator/rvc_models/`.

- `hubert_base/` — carpeta del modelo HuBERT, con `config.json` y los pesos (`model.safetensors` o `pytorch_model.bin`). Un `hubert_base.pt` suelto no alcanza.
- `rmvpe.pt` — el estimador de pitch.
- `tu-voz.pth` — el modelo de voz.
- `tu-voz.index` — opcional, el índice de esa voz.

Esos binarios quedan fuera de git. En la app: Actualizá los modelos y después Convertí la voz. El pitch es rmvpe; no hace falta pyworld.

`infer-rvc-python` pide `pyworld==0.3.4`, que no tiene wheel para Python 3.12. Si `pip install -r requirements-macos.txt` se cae ahí, instalá ese paquete con `--no-deps` después del resto. La app reemplaza pyworld por un stub y no lo llama.
