# Audio Separator

App local para Mac: separar voz e instrumental, bajar audio de YouTube, convertir voz con RVC en disco y volver a unir.

No usa el Space de Hugging Face.

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

Copiá estos archivos a `rvc_models/` antes de convertir. La app no los descarga.

- `hubert_base/` — carpeta del modelo HuBERT, con `config.json` y los pesos (`model.safetensors` o `pytorch_model.bin`). Un `hubert_base.pt` suelto no alcanza.
- `rmvpe.pt` — el estimador de pitch.
- `tu-voz.pth` — el modelo de voz.
- `tu-voz.index` — opcional, el índice de esa voz.

Esos binarios quedan fuera de git. En la app: Actualizá los modelos y después Convertí la voz. El pitch es rmvpe; no hace falta pyworld.

`infer-rvc-python` pide `pyworld==0.3.4`, que no tiene wheel para Python 3.12. Si `pip install -r requirements-macos.txt` se cae ahí, instalá ese paquete con `--no-deps` después del resto. La app reemplaza pyworld por un stub y no lo llama.
