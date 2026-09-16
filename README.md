# Audio Separator

App local para Mac **Intel** (o Apple Silicon con Rosetta): separar voz/instrumental, **entrenar y convertir voz con RVC**, y volver a unir. Todo en disco; sin subir audio a internet.

## Requisitos

- macOS 13+  
- CPU **x86_64** (Intel) o Rosetta 2  
- ~3 GB libres para el `.app` standalone completo  

## Ejecutable standalone

### Full (pesos adentro)

1. **Audio-Separator-macOS-Intel.zip** (~2 GB).  
2. Descomprimí → clic derecho → **Abrir**.  
3. Listo para Separar / Entrenar / Convertir.

### Lite (recomendado para descargas chicas)

1. **Audio-Separator-macOS-Intel-Lite.zip** (sin ~700 MB de pesos RVC).  
2. Abrí la app → **Completar instalación (pesos RVC)** (una vez; baja HuBERT, RMVPE, f0G/D desde Hugging Face público).  
3. Después Entrenar / Convertir funcionan offline.

Ambas: Mac **Intel** / Rosetta. No notarizado por Apple.

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
./scripts/build_standalone.sh       # full + zip
./scripts/build_standalone_lite.sh  # quita pesos RVC + zip lite
```

## Licencia

MIT. RVC-WebUI y modelos de terceros conservan sus licencias originales.
