# Audio Separator

App local para Mac **Intel** (o Apple Silicon con Rosetta): separar voz/instrumental, **entrenar y convertir voz con RVC**, y volver a unir. Todo en disco; sin subir audio a internet.

## Requisitos

- macOS 13+  
- CPU **x86_64** (Intel) o Rosetta 2  
- ~3 GB libres para el `.app` standalone completo  

## Ejecutable standalone (recomendado)

1. Descargá el release **Audio-Separator-macOS-Intel.zip** desde GitHub Releases.  
2. Descomprimí y mové `Audio Separator.app` a Aplicaciones (o donde quieras).  
3. La primera vez: clic derecho → **Abrir** (Gatekeeper; no está notarizado por Apple).  
4. Cerrá la ventana para salir.

Incluye: Python embebido, dependencias, RVC-WebUI, pesos de soporte (HuBERT, RMVPE, f0G/D40k).

## Flujo en la UI

1. **Canción** — archivo o YouTube  
2. **Extraer** — voz / instrumental  
3. **Resultado**  
4. **Voz (RVC)** — Entrenar → Convertir  
5. **Unir**  

Texto→habla (XTTS) es opcional y **no** viene con pesos; para clonar usá RVC.

## Desarrollo (repo)

```bash
cd /path/to/Audio_separator
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-macos.txt
# Colocá third_party/RVC-WebUI y library/models/rvc/ (ver abajo)
python desktop.py
# o: python app.py
```

### Pesos locales (`library/models/rvc/`)

| Archivo / carpeta | Uso |
| --- | --- |
| `hubert_base/` (Transformers: `config.json` + `model.safetensors`) | Entrenar |
| `hubert_base.pt` | Convertir (`infer_rvc_python`) |
| `rmvpe.pt` | Pitch |
| `f0G40k.pth` / `f0D40k.pth` | Base train |

Voces entrenadas: `library/models/rvc_voices/<nombre>.pth` (+ `.index`).

### CLI (con la app **cerrada**)

```bash
./convert_rvc.sh library/voices/voz_1.wav smoke_voice 0
```

No abras `desktop.py` y el CLI a la vez (segfault OpenMP en Intel).

### Empaquetar de nuevo

```bash
./scripts/build_standalone.sh
# → dist/Audio Separator.app
# → dist/Audio-Separator-macOS-Intel.zip
```

## Licencia

MIT. RVC-WebUI y modelos de terceros conservan sus licencias originales.
