# Atribución

El código de esta app (glue Gradio / desktop) es MIT (ver `LICENSE`).
Los modelos y bibliotecas de terceros **no** son MIT de Lucas Tomasi.

| Pieza | Uso | Origen |
|---|---|---|
| RVC-WebUI | Entrenar | Retrieval-based-Voice-Conversion-WebUI |
| `third_party/vc` | Convertir | Applio / RVC |
| HuBERT, RMVPE, f0G/D 40k | Entrenar / convertir | Hugging Face `lj1995/VoiceConversionWebUI` (gratis) |
| UVR MDX ONNX | Separar | TRvlvr `all_public_uvr_models` (gratis, GitHub Releases) |
| yt-dlp | YouTube | yt-dlp |
| Edge TTS | Texto → habla | Microsoft Edge (gratis, internet) |

Internet se usa para acelerar o para cuotas gratis. Si no hay red, lo que ya está en disco sigue andando.
