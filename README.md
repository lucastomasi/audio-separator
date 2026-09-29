# Audio Separator

App local para Mac: separar voz e instrumental, bajar audio de YouTube, convertir voz con RVC en disco y volver a unir.

No usa el Space de Hugging Face.

```bash
.venv/bin/python desktop.py
```

## Modelos RVC

Copiá estos archivos a `rvc_models/` antes de convertir. La app no los descarga.

- `hubert_base/` — carpeta del modelo HuBERT, con `config.json` y los pesos (`model.safetensors` o `pytorch_model.bin`). Un `hubert_base.pt` suelto no alcanza.
- `rmvpe.pt` — el estimador de pitch.
- `tu-voz.pth` — el modelo de voz.
- `tu-voz.index` — opcional, el índice de esa voz.

Esos binarios quedan fuera de git. En la app: Actualizá los modelos y después Convertí la voz. El pitch es rmvpe; no hace falta pyworld.

`infer-rvc-python` pide `pyworld==0.3.4`, que no tiene wheel para Python 3.12. Si `pip install -r requirements-macos.txt` se cae ahí, instalá ese paquete con `--no-deps` después del resto. La app reemplaza pyworld por un stub y no lo llama.
