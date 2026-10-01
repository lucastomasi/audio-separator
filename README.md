# Audio Separator

App local para Mac: separar voz e instrumental, bajar audio de YouTube, convertir voz con RVC en disco y volver a unir.

No usa el Space de Hugging Face.

```bash
.venv/bin/python desktop.py
```

## iPhone: agregarla a la pantalla de inicio

No es una app de la App Store ni un IPA. El teléfono es la pantalla. La separación y RVC corren en la computadora que tiene los modelos.

En esa computadora, en la misma Wi-Fi que el iPhone:

```bash
.venv/bin/python app.py --host 0.0.0.0
```

La terminal imprime una dirección `http://192.168.x.x:7860`. Abrila en Safari en el iPhone. `127.0.0.1` en el teléfono es el teléfono, no la computadora. `desktop.py` escucha solo en esta máquina y no alcanza para el iPhone.

En Safari: Compartir → Agregar a pantalla de inicio. El ícono abre la interfaz sin la barra del navegador.

Límites:

- Si apagás la computadora o cerrás el servidor, el ícono del iPhone no tiene a quién pedirle el audio.
- Los modelos ONNX y RVC no corren en el teléfono.
- Fuera de tu red no entra, salvo que publiques el servidor por tu cuenta. Esta app no trae hosting.
- En `http://` de la red local, Safari deja agregar a inicio. Un service worker de instalación offline pide HTTPS o localhost. Acá no hace falta para la pantalla de inicio, y el audio igual depende del servidor.
- Exportar a Descargas y Abrí la carpeta escriben en la computadora del servidor. Para guardar en el iPhone, usá la descarga del reproductor en Safari.

## Modelos RVC

Copiá estos archivos a `rvc_models/` antes de convertir. La app no los descarga.

- `hubert_base/` — carpeta del modelo HuBERT, con `config.json` y los pesos (`model.safetensors` o `pytorch_model.bin`). Un `hubert_base.pt` suelto no alcanza.
- `rmvpe.pt` — el estimador de pitch.
- `tu-voz.pth` — el modelo de voz.
- `tu-voz.index` — opcional, el índice de esa voz.

Esos binarios quedan fuera de git. En la app: Actualizá los modelos y después Convertí la voz. El pitch es rmvpe; no hace falta pyworld.

`infer-rvc-python` pide `pyworld==0.3.4`, que no tiene wheel para Python 3.12. Si `pip install -r requirements-macos.txt` se cae ahí, instalá ese paquete con `--no-deps` después del resto. La app reemplaza pyworld por un stub y no lo llama.
