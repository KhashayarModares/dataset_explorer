# RIR Database — reproducible metadata explorer and acoustic analysis

A research-oriented Python workflow for cataloguing **room impulse response (RIR)** collections in SQLite, estimating selected acoustic descriptors, and exploring results in a Streamlit dashboard.

**Scope.** The repository contains application code, ingestion tools, and analysis scripts. It is **not** a complete distribution of the original RIR recordings. Raw WAV, SOFA, and MAT files and the full institutional metadata database must be obtained separately, subject to each dataset's access and license terms.

## 1. System architecture

| Component | Entry point | Responsibility |
| --- | --- | --- |
| SQLite | local \`rir_meta_v3.db\` | Persistent \`rirs\` table: one metadata record per ingested path |
| SQLAlchemy + pandas | \`app/app.py\` | Read/query SQLite and populate filtered pandas tables |
| Streamlit + Altair | \`app/app.py\` | Browser interface, filters, plots, CSV export, local-file ZIP exports |
| SQLite ingester | \`tools/ingest_bras_audio.py\` | Scan WAV/SOFA/MAT files and insert/update their metadata |
| Acoustic analysis | \`scripts/backfill_t60.py\` | Estimate T60 and C50/C80 from supported impulse response files |
| Boundary-point proxy | \`scripts/backfill_bp.py\` | Populate a log10 descriptor from available metadata (see limitations) |

Workflow: original recordings → ingester → SQLite → acoustic backfill → SQLite → SQLAlchemy queries → Streamlit results. **SQLAlchemy is the database connection/query layer; it does not store the audio.** Streamlit runs a local web server; the browser is only the interface, while file access and SQL queries run on the host executing Streamlit.

## 2. Repository contents

~~~text
dataset_explorer/
├── app/app.py                   # Streamlit dashboard
├── tools/ingest_bras_audio.py   # WAV/SOFA/MAT metadata ingester
├── scripts/backfill_t60.py     # Reverberation and clarity estimates
├── scripts/backfill_bp.py      # Boundary-point / proxy descriptor
├── requirements.txt            # Dashboard dependencies
├── dev-requirements.txt        # Audio/scientific dependencies
├── runtime.txt                 # Python 3.11 for hosted environments
├── data/public_sample/         # Small illustrative files; NOT the full corpus
└── README.md
~~~

Recommended layout on your own machine (**outside the Git checkout**):

~~~text
$HOME/
├── code/dataset_explorer/                 # This Git repository
└── Database/
    ├── db/rir_meta_v3.db                  # Your existing metadata SQLite file
    ├── data/
    │   ├── BRAS/
    │   ├── AnotherDataset/
    │   └── MyNewDataset/                 # Put extracted WAV/SOFA/MAT files here
    └── tmp/                              # Optional temporary ZIP workspace
~~~

Any readable paths are acceptable when consistently supplied in the commands. Do not commit institutional datasets, private paths, credentials, a large SQLite database, or copyrighted audio into Git. Back up the SQLite file before running any script that writes to it.

## 3. Installation (Linux / WSL / macOS)

Python **3.11** is recommended for the pinned scientific packages.

~~~bash
git clone https://github.com/KhashayarModares/dataset_explorer.git
cd dataset_explorer
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r dev-requirements.txt

mkdir -p "$HOME/Database/db" "$HOME/Database/data" "$HOME/Database/tmp"
export RIR_DB="$HOME/Database/db/rir_meta_v3.db"
export RIR_DB_PATH="$RIR_DB"
export RIR_DATA_ROOT="$HOME/Database/data"
export TMPDIR="$HOME/Database/tmp"
export RIR_ALLOW_OUTSIDE=0
unset RIR_PGURL
~~~

**Important:** the preceding commands create folders, **not** a populated database. Copy your existing \`rir_meta_v3.db\` into the specified \`db/\` folder, or follow §6 to create a new catalog by ingestion. The application requires the \`rirs\` table.

A verified existing lab layout is:

~~~bash
cd /home/on61ewex/code/dataset_explorer
source .venv/bin/activate
export RIR_DB="/home/on61ewex/Database/db/rir_meta_v3.db"
export RIR_DB_PATH="$RIR_DB"
export RIR_DATA_ROOT="/home/on61ewex/Database/data"
export TMPDIR="/home/on61ewex/Database/tmp"
mkdir -p "$TMPDIR"
~~~

### Windows PowerShell (local installation)

~~~powershell
git clone https://github.com/KhashayarModares/dataset_explorer.git
cd dataset_explorer
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r dev-requirements.txt

# Replace these with the real paths on your Windows laptop:
$env:RIR_DB = "C:\Users\Khashayar\dataset_explorer\data\rir_meta_v3.db"
$env:RIR_DB_PATH = $env:RIR_DB
$env:RIR_DATA_ROOT = "C:\Users\Khashayar\RIR_Data"
$env:RIR_ALLOW_OUTSIDE = "0"
~~~

If you downloaded the SQLite file without a \`.db\` extension, point \`RIR_DB\` at its **actual filename**, or rename the file to \`rir_meta_v3.db\`. Do **not** use a placeholder such as \`C:\path\to\...\`: SQLite can otherwise create an empty file or fail to open it. Linux paths such as \`/home/on61ewex/...\` cannot be accessed by a native Windows process.

## 4. Validate the database before opening the app

Linux (requires the \`sqlite3\` command-line utility):

~~~bash
test -f "$RIR_DB" || { echo "Database file not found: $RIR_DB"; exit 1; }
sqlite3 "$RIR_DB" "SELECT name FROM sqlite_master WHERE type='table' AND name='rirs';"
sqlite3 "$RIR_DB" "SELECT COUNT(*) AS rir_records FROM rirs;"
sqlite3 "$RIR_DB" "SELECT dataset, COUNT(*) FROM rirs GROUP BY dataset ORDER BY dataset LIMIT 20;"
~~~

Cross-platform Python verification, from the activated environment:

~~~bash
python -c "import os,sqlite3,pathlib; p=pathlib.Path(os.environ['RIR_DB']); assert p.is_file(), f'Missing DB: {p}'; c=sqlite3.connect(f'file:{p.as_posix()}?mode=ro',uri=True); print('tables:',c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()); print('records:',c.execute('SELECT COUNT(*) FROM rirs').fetchone()[0]); c.close()"
~~~

If \`rirs\` is absent, you are using an empty/wrong SQLite file. **Do not regenerate the real catalog from the dashboard.** Restore the correct file, or ingest files as described below.

## 5. Run and use the Streamlit dashboard

From the **repository root**, with the virtual environment and variables from §3 active:

~~~bash
python -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501
~~~

Open **http://127.0.0.1:8501** in the browser on the same computer. On Windows use the same \`python -m streamlit ...\` command in PowerShell. The terminal remains occupied while Streamlit runs; this is normal. Press **Ctrl+C** to stop it.

How to use it:

1. Choose datasets, file types, sample rates, room/LOS and other optional filters in the sidebar.
2. Click **Search**; the app queries the \`rirs\` table using SQLAlchemy and pandas.
3. Inspect the table and metric distributions. Pagination means exports refer to the **displayed results page**, not automatically to the entire database.
4. Use **Download CSV** for metadata. For recordings, click **Build ZIP now**, check the number of copied files, then **Download ZIP**.

A ZIP includes \`manifest.csv\` and \`zip_diagnostics.txt\`. Recordings are included **only when their paths resolve on the computer running Streamlit**. Downloading only the SQLite database does not download the original recordings. If many paths are listed as \`MISSING\`, provide the original data tree or set \`RIR_PATH_MAP\` (see §9). For large datasets, start with a small result page: ZIP creation may require substantial RAM, disk space, and time. Treat the \`_exports\` copy and diagnostics as potentially sensitive; restrict access if using shared machines.

**Networking:** \`127.0.0.1\` is the machine running the app; it is not a remote server address. For remote access, prefer an authenticated SSH port forward or a properly access-controlled tunnel. Do not publicly expose an unrestricted dashboard connected to private datasets. GitHub hosts the **source code**; cloning from GitHub does not make the lab server's database available.

## 6. Add a new dataset (worked example)

Assume you have obtained permission to use a new dataset, and the extracted files are stored at:

~~~text
$HOME/Database/data/MyNewDataset/
├── room_A/ir_001.wav
├── room_A/ir_002.wav
└── room_B/ir_003.wav
~~~

The ingester scans files ending in **.wav, .sofa, or .mat**. It reads available headers and dataset-specific metadata. Some fields (e.g., physical room geometry, line of sight, source/receiver positions) cannot be inferred reliably from an arbitrary WAV filename and may remain NULL. Naming/layout-specific readers in the script support particular corpora; this is **not** a universal metadata parser.

From the repository root:

~~~bash
source .venv/bin/activate
export RIR_DB="$HOME/Database/db/rir_meta_v3.db"
export RIR_DATA_ROOT="$HOME/Database/data"

# Inspect what the script accepts:
python tools/ingest_bras_audio.py --help

# Small initial test: at most 10 files
python tools/ingest_bras_audio.py \
  --db "$RIR_DB" \
  --root "$RIR_DATA_ROOT/MyNewDataset" \
  --dataset "MyNewDataset" \
  --max 10

# Inspect the results before importing the full collection
sqlite3 "$RIR_DB" "SELECT file_name, sample_rate_hz, duration_s FROM rirs WHERE dataset='MyNewDataset' LIMIT 10;"

# Import all matching files (0 means unlimited)
python tools/ingest_bras_audio.py \
  --db "$RIR_DB" \
  --root "$RIR_DATA_ROOT/MyNewDataset" \
  --dataset "MyNewDataset" \
  --glob '**/*.*' \
  --max 0
~~~

The script creates/extends the \`rirs\` schema and indexes if needed, and upserts records by \`file_path\`. Re-running on the same resolved file paths should update those records rather than deliberately creating a second dataset copy; verify counts after ingestion. Use a **copy** of a valuable database when experimenting with an unfamiliar corpus.

For a fresh, dedicated database, set \`RIR_DB\` to a new path in an existing writable directory, then run the ingester **first**. The app itself does not create the schema.

## 7. Calculate T60, C50 and C80

The script \`scripts/backfill_t60.py\` reads audio samples (including supported WAV, SOFA, MAT and certain archive/HDF5 representations), computes a backward-integrated energy decay curve (Schroeder-style), and estimates reverberation time by a linear fit over the decay range. It first attempts a **T30** fit (-5 to -35 dB), then a **T20** fallback (-5 to -25 dB) scaled to an equivalent 60 dB decay. Estimates may fail when signals are too short, silent or inadequately decaying.

Clarity is computed from the ratio of early to late squared-sample energy:

\`C50 = 10 log10(E[0–50 ms] / E[50 ms–end])\`

\`C80 = 10 log10(E[0–80 ms] / E[80 ms–end])\`

The current implementation computes these from a chosen/averaged one-dimensional response and stores them as \`c50_db\` and \`c80_db\`; results depend on which channel/measurement is selected by the loader. This is an exploratory estimator, **not** a substitute for validating signal onset, noise floor, channel selection and ISO 3382 measurement conditions for a formal acoustic study.

~~~bash
# Analyze up to 20 rows without T60 (first test)
python scripts/backfill_t60.py --db "$RIR_DB" --dataset "MyNewDataset" --limit 20

# Analyze up to 100000 rows with missing T60
python scripts/backfill_t60.py --db "$RIR_DB" --dataset "MyNewDataset" --limit 100000

# Also revisit rows that already have T60 (e.g. to populate newly added clarity columns)
python scripts/backfill_t60.py --db "$RIR_DB" --dataset "MyNewDataset" \
  --include-existing-t60 --limit 100000

# Validate the results
sqlite3 "$RIR_DB" "SELECT file_name, t60_s, t60_method, c50_db, c80_db FROM rirs WHERE dataset='MyNewDataset' LIMIT 20;"
~~~

By default the backfill processes rows with \`t60_s IS NULL\`. The optional \`--include-existing-t60\` flag reruns acoustic estimation on existing-T60 rows too; use it only when overwriting those estimates is acceptable. The script reports per-file failures and records T60 errors in \`t60_method\`. Missing files, unsupported containers or non-RIR signals can produce null values. **Inspect a sample of results before applying any published statistics.**

## 8. Optional boundary-point/proxy descriptor

~~~bash
python scripts/backfill_bp.py --db "$RIR_DB"
sqlite3 "$RIR_DB" "SELECT dataset, bp_log10, bp_source FROM rirs LIMIT 10;"
~~~

**Interpretation warning:** when true \`boundary_points\` are unavailable, this script falls back to log10(volume), log10(number of receivers), or log10(distance), recording the choice in \`bp_source\`. These proxies are **not actual boundary-point counts** and should not be mixed as a physically equivalent measurement in research plots.

## 9. Data portability and download troubleshooting

The \`file_path\` column may hold absolute Linux paths from the original lab. After moving the SQLite file to Windows, those paths do not automatically become Windows paths.

The app accepts these environment settings:

| Variable | Meaning |
| --- | --- |
| \`RIR_DB\`, \`RIR_DB_PATH\` | SQLite database filename; \`RIR_DB_PATH\` takes precedence |
| \`RIR_DATA_ROOT\` | Root directory of downloaded recordings |
| \`RIR_PATH_MAP\` | Semicolon-separated old-prefix=new-prefix mappings for file lookup |
| \`RIR_ALLOW_OUTSIDE\` | Set to \`1\` only if exporting files outside \`RIR_DATA_ROOT\` is intentional |
| \`TMPDIR\` | Writable staging directory for ZIP creation |
| \`RIR_EXPORTS\` | Optional folder for a copy of the generated ZIP |
| \`PUBLIC_DATA_BASE_URL\` | Optional externally served base URL; does not upload files |

For example, when a Linux-stored dataset is copied to Windows:

~~~powershell
$env:RIR_DB = "C:\Users\Khashayar\dataset_explorer\data\rir_meta_v3.db"
$env:RIR_DB_PATH = $env:RIR_DB
$env:RIR_DATA_ROOT = "D:\RIR_Data"
$env:RIR_PATH_MAP = "/home/on61ewex/Database/data/=D:\RIR_Data\"
python -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501
~~~

The path mapping only helps when the corresponding files truly exist on the local disk. If you are using the original Linux data tree, run Streamlit on that machine and access its interface securely from your laptop; this avoids copying the complete corpus.

Typical errors:

- **\`no such table: rirs\`** — wrong/empty SQLite file; verify §4.
- **\`unable to open database file\`** — nonexistent path, missing parent directory or insufficient permissions.
- **\`streamlit: command not found\`** — activate the correct environment or use \`python -m streamlit\`.
- **\`ERR_ADDRESS_INVALID\`** — browse to \`127.0.0.1\`, not \`0.0.0.0\`.
- **ZIP includes only manifest** — path mismatch, missing source files, or outside-root policy; read \`zip_diagnostics.txt\`.
- **Very large export stalls** — reduce the number of rows; ZIPs are constructed in memory in this implementation.
- **Changes do not appear** — rerun Search; some filter option queries are cached for up to five minutes.

## 10. Research reproducibility and responsible data use

For a publication or dataset release, record:

1. Dataset names, original authors, URLs, release versions, licenses, and access dates.
2. SHA-256 digests of original archives and SQLite snapshots; do not silently change source audio.
3. The Git commit hash (\`git rev-parse HEAD\`), Python version, dependency versions (\`python -m pip freeze\`), and exact commands used.
4. Calibration assumptions, response/channel selection, fit intervals, invalid-signal handling, missing-value counts, and any dataset-specific metadata parsing.
5. Whether fields represent direct measurements, estimated parameters, or proxies; keep \`t60_method\` and \`bp_source\` alongside reported values.

Quick quality checks:

~~~bash
sqlite3 "$RIR_DB" "PRAGMA integrity_check;"
sqlite3 "$RIR_DB" "SELECT dataset, COUNT(*), SUM(t60_s IS NOT NULL), SUM(c50_db IS NOT NULL), SUM(c80_db IS NOT NULL) FROM rirs GROUP BY dataset;"
sha256sum "$RIR_DB"   # Linux/WSL
git rev-parse HEAD
python --version
~~~

**Citing this software:** cite the repository URL and the exact commit/release you used, as well as the original RIR datasets and any standards/algorithms relevant to your analysis. No DOI, formal publication, software license, or research affiliation is asserted here without a corresponding record in this repository. Copyright and redistribution conditions of third-party recordings remain with their owners.

## 11. Project maintenance

- Edit scripts in a feature branch; test against a copy of the database.
- Verify \`python -m py_compile app/app.py tools/ingest_bras_audio.py scripts/backfill_t60.py scripts/backfill_bp.py\`.
- Keep generated databases and raw archives out of commits. Review \`git status\` before pushing.
- The dataset-specific ingester and acoustic loaders should be extended and validated when introducing a new file format or corpus structure.

**Maintainer:** [KhashayarModares](https://github.com/KhashayarModares). Issues and reproducible bug reports: [repository issues](https://github.com/KhashayarModares/dataset_explorer/issues).
