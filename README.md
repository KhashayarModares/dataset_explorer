# dataset_explorer

Public, single-branch repo containing:
- `app/app.py` — Streamlit UI to browse/search room impulse response (RIR) metadata in a SQLite DB
- `scripts/backfill_t60.py` — compute RIR metrics (T60 currently; extendable for C50/C80/etc.)
- `tools/ingest_bras_audio.py` — unified ingester for WAV/SOFA/MAT (+ dataset-specific helpers)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# environment
export RIR_DB_PATH="/path/to/rir_meta_v3.db"
export RIR_DATA_ROOT="/path/to/data"    # folder that contains your datasets

# run the UI
streamlit run app/app.py

