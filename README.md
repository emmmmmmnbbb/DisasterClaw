# DisasterClaw

Single-UAV disaster response console built around:

- real satellite basemap (Esri World Imagery / OSM) via Leaflet
- xBD disaster imagery overlaid on its true georeferenced footprint
- per-tile building annotations with damage-level coloring
- mock UAV with default 30m hover, NED frame re-anchored to each active tile
- AI task planning with AerialClaw-style LLM config and rule fallback, aware of the active tile bbox
- vision analysis via OpenAI-compatible VLM or rs_agent_system-style local Qwen2.5-VL loading

## Structure

- `backend/`: Flask + Socket.IO backend
- `frontend/`: React + Vite frontend
- `docs/`: project notes
- `scripts/`: local run helpers
- `paper/`: CVPR-style DisasterClaw manuscript, claim contract, and strict
  experiment protocol

## Paper

The anonymous English draft is built from traceable benchmark artifacts:

```bash
python scripts/benchmarks/export_paper_assets.py
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error paper/main.tex
```

See `paper/CLAIMS.md` for the allowed scientific claims and
`paper/EXPERIMENT_PROTOCOL.md` for the event-disjoint evaluation protocol.

## Default Anchor

The anchor is chosen dynamically from the xBD manifest on startup:

1. First georeferenced tile of `XBD_DEFAULT_DISASTER` / `XBD_DEFAULT_STAGE`
   (defaults `hurricane-florence` / `post`).
2. Any tile with `has_georef: true`.
3. Fallback to Shanghai `31.2304, 121.4737` if no manifest is available.

The UAV always starts directly above the anchor at `30m`. Whenever a new xBD
tile is activated from the UI, the world anchor snaps to that tile's center
and all NED offsets are recomputed.

## xBD disaster map

- Base layer: Esri World Imagery (OSM / Carto Dark selectable in the layer
  control).
- The active xBD tile is rendered as a georeferenced `ImageOverlay`, with the
  opacity slider in the HUD.
- `footprints.geojson` is drawn as clickable polygons - click a footprint to
  activate that tile.
- Per-tile annotations are fetched from `/api/xbd/annotations/<tile_id>` and
  colored by damage level.
- Mouse hover shows lat/lon plus point elevation (Open-Meteo, cached via
  `/api/elevation`).
- Click the map to set a cursor target; `Fly` sends `fly_to_geo`, `AI Inspect`
  delegates to the planner with the clicked lat/lon.

### Dataset layout

The backend reads the manifest from `backend/data/xbd/manifest.json` and tile
footprints from `backend/data/xbd/footprints.geojson`. They can be produced
by running `AerialClaw`'s xBD pipeline, or symlinked to an existing artifact:

```bash
mkdir -p /home/lc/disasterclaw/backend/data/xbd
ln -s /home/lc/AerialClaw/data/xbd/manifest.json \
      /home/lc/disasterclaw/backend/data/xbd/manifest.json
ln -s /home/lc/AerialClaw/data/xbd/footprints.geojson \
      /home/lc/disasterclaw/backend/data/xbd/footprints.geojson
```

Raw `images/*.png` and `labels/*.json` are read from `XBD_DATASET_ROOT`
(defaults to `~/datasets/xbd`). Override with `XBD_MANIFEST_PATH`,
`XBD_FOOTPRINTS_PATH`, `XBD_DATASET_ROOT` when your layout differs.

## LLM Configuration

Use the same pattern as AerialClaw:

```bash
cd /home/lc/disasterclaw
cp .env.example .env
```

Then edit `.env` and set:

- `ACTIVE_PROVIDER`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `VLM_BASE_URL`
- `VLM_API_KEY`
- `VLM_MODEL`
- optional `BASE_MODEL` / `CHECKPOINT_PATH` when using local `qwen_vl_local`
- optional `PLANNER_LLM_PROVIDER` / `PLANNER_LLM_MODEL`

You can inspect the resolved runtime config from:

```bash
curl http://127.0.0.1:5011/api/llm/config
```

## Conda Environment

The project now has a dedicated Conda environment:

```bash
conda activate disasterclaw
```

If you need to recreate it later:

```bash
cd /home/lc/disasterclaw
conda env create -f environment.yml
```

The backend launcher [`run_backend.sh`](scripts/run_backend.sh) will prefer:

- `DISASTERCLAW_PYTHON_BIN`
- current active Conda env
- `/home/lc/miniconda3/envs/disasterclaw/bin/python`
- fallback `AerialClaw/.venv`

## Qwen2-VL / Qwen2.5-VL

The backend now exposes a dedicated VLM route:

```bash
POST /api/vlm/analyze
```

The web UI includes a `Vision Upload` panel, so you can upload a local image from your browser and send it either to:

- an OpenAI-compatible VLM endpoint
- a locally loaded Qwen2.5-VL backend using the same style as `rs_agent_system`

Typical local Ollama setup:

```bash
ollama serve
ollama pull qwen2.5vl:7b
```

Then put this in `.env`:

```dotenv
VLM_PROVIDER=vlm
VLM_BASE_URL=http://127.0.0.1:11434/v1
VLM_API_KEY=ollama
VLM_MODEL=qwen2.5vl:7b
VLM_IMAGE_INPUT_MODE=auto
```

If your service already exposes a different model name such as `qwen2vl`, keep the same URL and replace only `VLM_MODEL`.

### Local Transformers backend (rs_agent_system style)

If you already have the model on disk and want to avoid `ollama`, switch the provider:

```dotenv
VLM_PROVIDER=qwen_vl_local
BASE_MODEL=/path/to/Qwen2.5-VL-7B-Instruct
CHECKPOINT_PATH=
VLM_LOCAL_DEVICE=auto
VLM_LOCAL_TORCH_DTYPE=auto
VLM_LOCAL_MIN_FREE_GPU_GB=12
VLM_LOCAL_TOP_P=0.9
VLM_LOCAL_REPETITION_PENALTY=1.1
```

This backend follows the same idea as [`rs_agent_system`](../rs_agent_system):

- `transformers.AutoProcessor`
- `AutoModelForImageTextToText`
- optional `peft` LoRA merge
- `model.generate()` for final inference

If you also want the planner to use the same local backend, set:

```dotenv
PLANNER_LLM_PROVIDER=qwen_vl_local
```

Local mode requires these Python packages in the runtime environment:

```bash
pip install pillow transformers accelerate peft
```

## Run

Backend:

```bash
cd /home/lc/disasterclaw
conda activate disasterclaw
./scripts/run_backend.sh
```

Frontend dev server:

```bash
cd /home/lc/disasterclaw
./scripts/run_frontend.sh
```

The frontend proxies `/api` and `/socket.io` to the backend on `127.0.0.1:5011`.
Basemap tiles (`server.arcgisonline.com`, `tile.openstreetmap.org`,
`basemaps.cartocdn.com`) are requested directly by the browser - make sure
outbound HTTPS is available. For fully offline use, set
`XBD_ELEVATION_DISABLE=1` and serve your own basemap via `XBD_BASEMAP`.

## Recommended Remote Usage

On the remote server:

```bash
cd /home/lc/disasterclaw
cp .env.example .env
./scripts/build_frontend.sh
./scripts/run_backend.sh
```

On your local machine, create an SSH tunnel:

```bash
ssh -N -L 5011:127.0.0.1:5011 -p <SSH_PORT> <USER>@<REMOTE_HOST>
```

Then open locally:

```text
http://127.0.0.1:5011
```

After the page opens, use the `Vision Upload` panel to validate Qwen2-VL directly from your local browser.

You can also use the helper script on your local machine:

```bash
export DISASTERCLAW_REMOTE_HOST=<REMOTE_HOST>
export DISASTERCLAW_REMOTE_USER=<USER>
export DISASTERCLAW_REMOTE_PORT=<SSH_PORT>
./scripts/start_web_tunnel.sh
```
