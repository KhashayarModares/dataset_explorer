# RIR Database — reproducible metadata explorer and acoustic analysis

A research-oriented Python workflow for cataloguing **room impulse response (RIR)** collections in SQLite, estimating selected acoustic descriptors, and exploring results in a Streamlit dashboard.

**Scope.** The repository contains application code, ingestion tools, and analysis scripts. It is **not** a complete distribution of the original RIR recordings. Raw WAV, SOFA, and MAT files and the full institutional metadata database must be obtained separately, subject to each dataset's access and license terms.

## 1. System architecture

| Component | Entry point | Responsibility |
| --- | --- | --- |
| SQLite | local `rir_meta_v3.db` | Persistent `rirs` table: one metadata record per ingested path |
| SQLAlchemy + pandas | `app/app.py` | Read/query SQLite and populate filtered pandas tables |
| Streamlit + Altair | `app/app.py` | Browser interface, filters, plots, CSV export, local-file ZIP exports |
| SQLite ingester | `tools/ingest_bras_audio.py` | Scan WAV/SOFA/MAT files and insert/update their metadata |
| Acoustic analysis | `scripts/backfill_t60.py` | Estimate T60 and C50/C80 from supported impulse response files |
| Boundary-point proxy | `scripts/backfill_bp.py` | Populate a log10 descriptor from available metadata (see limitations) |

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
├── app/export_utils.py         # Size preflight, byte-progress ZIP and copy operations
├── tests/test_export_utils.py  # Offline regression tests
├── CITATION.cff                # Citation metadata
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
    └── tmp/                              # Default root for uncompressed file copies
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
export RIR_COPY_ROOT="$TMPDIR"  # Restrict destination choices to this host directory
unset RIR_PGURL
~~~

**Important:** the preceding commands create folders, **not** a populated database. Copy your existing `rir_meta_v3.db` into the specified `db/` folder, or follow §6 to create a new catalog by ingestion. The application requires the `rirs` table.

For the maintainer's existing Linux installation paths, see §9.2. Other users should configure their own data locations.

### Windows PowerShell (local installation)

~~~powershell
git clone https://github.com/KhashayarModares/dataset_explorer.git
cd dataset_explorer
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r dev-requirements.txt

# Replace these with the real paths on your Windows laptop:
$env:RIR_DB = Join-Path $HOME "dataset_explorer\data\rir_meta_v3.db"
$env:RIR_DB_PATH = $env:RIR_DB
$env:RIR_DATA_ROOT = Join-Path $HOME "RIR_Data"
$env:RIR_ALLOW_OUTSIDE = "0"
~~~

If you downloaded the SQLite file without a `.db` extension, point `RIR_DB` at its **actual filename**, or rename the file to `rir_meta_v3.db`. Do **not** use a placeholder such as `C:\path\to\...`: SQLite can otherwise create an empty file or fail to open it. Linux paths are not accessible to a native Windows process unless the files are mounted or transferred.

## 4. Validate the database before opening the app

Linux (requires the `sqlite3` command-line utility):

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

If `rirs` is absent, you are using an empty/wrong SQLite file. **Do not regenerate the real catalog from the dashboard.** Restore the correct file, or ingest files as described below.

## 5. Run and use the Streamlit dashboard

From the **repository root**, with the virtual environment and variables from §3 active:

~~~bash
python -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501
~~~

Open **http://127.0.0.1:8501** in the browser on the same computer. On Windows use the same `python -m streamlit ...` command in PowerShell. The terminal remains occupied while Streamlit runs; this is normal. Press **Ctrl+C** to stop it.

How to use it:

1. Choose datasets, file types, sample rates, room/LOS and other optional filters in the sidebar.
2. Click **Search**; the app queries the `rirs` table using SQLAlchemy and pandas.
3. Inspect the table and metric distributions. Pagination means exports refer to the **displayed results page**, not automatically to the entire database.
4. Use **Download CSV** for metadata. For recordings, click **Prepare ZIP**. The app first scans accessible source files, reports their count and total size, and asks for confirmation if the size exceeds **1 GB (1,000,000,000 bytes)**. During compression, a progress bar reports source bytes processed, elapsed time, and an approximate remaining time. Finally click **Download ZIP**.
5. To avoid compression, enter a destination under the configured **RIR_COPY_ROOT** and click **Copy selected files (no compression)**. The app copies original files into a new uniquely named `rir_copy_*` folder **on the Streamlit host**, preserving dataset subfolders. A progress bar shows the number of bytes copied.

ZIPs and uncompressed copy folders include `manifest.csv` and `zip_diagnostics.txt`. Files inside either export are under a `files/` subfolder. Recordings are included **only when their paths resolve on the computer running Streamlit**. Downloading only the SQLite database does not download the original recordings. If many paths are listed as `MISSING`, provide the original data tree or set `RIR_PATH_MAP` (see §9). For large datasets, start with a small result page: ZIP creation may require substantial RAM, disk space, and time. Streamlit's browser download button retains the ZIP data in memory, so very large archives can still exhaust memory even though compression now streams input in chunks. The uncompressed host-copy option avoids archive creation and browser download. Treat the `_exports` copy and diagnostics as potentially sensitive; restrict access if using shared machines.

**Networking:** `127.0.0.1` is the machine running the app; it is not a remote server address. For remote access, prefer an authenticated SSH port forward or a properly access-controlled tunnel. Do not publicly expose an unrestricted dashboard connected to private datasets. GitHub hosts the **source code**; cloning it does not copy external metadata databases or source recordings.

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

The script creates/extends the `rirs` schema and indexes if needed, and upserts records by `file_path`. Re-running on the same resolved file paths should update those records rather than deliberately creating a second dataset copy; verify counts after ingestion. Use a **copy** of a valuable database when experimenting with an unfamiliar corpus.

For a fresh, dedicated database, set `RIR_DB` to a new path in an existing writable directory, then run the ingester **first**. The app itself does not create the schema.

## 7. Calculate T60, C50 and C80

The script `scripts/backfill_t60.py` reads audio samples (including supported WAV, SOFA, MAT and certain archive/HDF5 representations), computes a backward-integrated energy decay curve (Schroeder-style), and estimates reverberation time by a linear fit over the decay range. It first attempts a **T30** fit (-5 to -35 dB), then a **T20** fallback (-5 to -25 dB) scaled to an equivalent 60 dB decay. Estimates may fail when signals are too short, silent or inadequately decaying.

Clarity is computed from the ratio of early to late squared-sample energy:

`C50 = 10 log10(E[0–50 ms] / E[50 ms–end])`

`C80 = 10 log10(E[0–80 ms] / E[80 ms–end])`

The current implementation computes these from a chosen/averaged one-dimensional response and stores them as `c50_db` and `c80_db`; results depend on which channel/measurement is selected by the loader. This is an exploratory estimator, **not** a substitute for validating signal onset, noise floor, channel selection and ISO 3382 measurement conditions for a formal acoustic study.

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

By default the backfill processes rows with `t60_s IS NULL`. The optional `--include-existing-t60` flag reruns acoustic estimation on existing-T60 rows too; use it only when overwriting those estimates is acceptable. The script reports per-file failures and records T60 errors in `t60_method`. Missing files, unsupported containers or non-RIR signals can produce null values. **Inspect a sample of results before applying any published statistics.**

## 8. Optional boundary-point/proxy descriptor

~~~bash
python scripts/backfill_bp.py --db "$RIR_DB"
sqlite3 "$RIR_DB" "SELECT dataset, bp_log10, bp_source FROM rirs LIMIT 10;"
~~~

**Interpretation warning:** when true `boundary_points` are unavailable, this script falls back to log10(volume), log10(number of receivers), or log10(distance), recording the choice in `bp_source`. These proxies are **not actual boundary-point counts** and should not be mixed as a physically equivalent measurement in research plots.

## 9. Deployment and portability

The application accesses the **SQLite database and the original audio files on the machine running Streamlit**. The database stores metadata and paths; it does not contain the full WAV/SOFA/MAT corpus. A metadata-only installation supports searching, plotting, and CSV export, but it cannot create ZIP archives of recordings that are not locally accessible.

### 9.1 Public setup — independent workstation or research server

This procedure uses only generic paths. Adjust them for your own installation; **do not copy example paths literally**.

**Files required**

| Item | Suggested location | Purpose |
| --- | --- | --- |
| This Git repository | `$HOME/code/dataset_explorer/` | UI and processing scripts |
| Populated SQLite file | `$HOME/Database/db/rir_meta_v3.db` | Metadata query and export |
| Dataset recordings | `$HOME/Database/data/<dataset>/` | Download of audio and acoustic backfill |
| Python environment | `dataset_explorer/.venv/` | Install independently on each operating system |
| Temporary/export folders | `$HOME/Database/tmp/`, `$HOME/Database/data/_exports/` | Uncompressed host copies and optional saved ZIPs |

**Run the app on Linux** after cloning the repository and supplying a real, populated database:

~~~bash
cd "$HOME/code/dataset_explorer"
source .venv/bin/activate
export RIR_DB="$HOME/Database/db/rir_meta_v3.db"
export RIR_DB_PATH="$RIR_DB"
export RIR_DATA_ROOT="$HOME/Database/data"
export TMPDIR="$HOME/Database/tmp"
export RIR_EXPORTS="$RIR_DATA_ROOT/_exports"
export RIR_COPY_ROOT="$TMPDIR"
export RIR_ALLOW_OUTSIDE=0
unset RIR_PATH_MAP
mkdir -p "$TMPDIR"
test -r "$RIR_DB"
sqlite3 "$RIR_DB" "SELECT COUNT(*) FROM rirs;"
python -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501
~~~

Open **http://127.0.0.1:8501** on that computer. The terminal is occupied while Streamlit runs; stop it with Ctrl+C.

**Move the app to another machine (Linux, Windows, or macOS):**

1. Clone the repository again on the destination. Copying only `app.py` is insufficient: keep `requirements.txt`, `dev-requirements.txt`, `tools/`, `scripts/` and this documentation. Do **not** copy a virtual environment between operating systems.
2. Copy the SQLite database using a consistent backup. For a database that may be in SQLite WAL mode, back up from its source using:

   ~~~bash
   mkdir -p "$HOME/rir-transfer"
   sqlite3 "$RIR_DB" ".backup '$HOME/rir-transfer/rir_meta_v3.db'"
   sqlite3 "$HOME/rir-transfer/rir_meta_v3.db" "PRAGMA integrity_check;"
   ~~~

   Transfer the backup with an approved file-transfer method. The integrity check should print `ok`. Do not transfer only an actively written main `.db` file without accounting for WAL state.
3. If the destination must support ZIP download of original recordings or acoustic recomputation, also transfer the **actual data directory**, preserving dataset subdirectories. For large collections, use a resumable transfer tool and verify sizes/checksums. If only CSV metadata and visualization are required, copying recordings is optional.
4. Recreate the virtual environment and install the dependencies described in §3. Set `RIR_DB`, `RIR_DB_PATH`, and `RIR_DATA_ROOT` to the destination's **real** paths.
5. Check path resolution and ZIP output using a small filtered search before starting a large export.

For a dataset tree moved to a different Linux location, remap stored prefixes in the **app** as follows:

~~~bash
export RIR_DB="/new/workspace/db/rir_meta_v3.db"
export RIR_DB_PATH="$RIR_DB"
export RIR_DATA_ROOT="/new/workspace/data"
export RIR_PATH_MAP="/old/workspace/data/=/new/workspace/data/"
export RIR_ALLOW_OUTSIDE=0
~~~

This works only if the old prefix matches `rirs.file_path` and the corresponding new files exist. **Path remapping applies to the Streamlit file lookup, not to every analysis script.** For `backfill_t60.py`, ensure SQLite paths are valid on the host performing analysis, or update paths carefully in a backed-up copy of the database.

For **Windows PowerShell**, after copying the database and, if needed, recordings:

~~~powershell
cd "$HOME\dataset_explorer"
.\.venv\Scripts\Activate.ps1
$env:RIR_DB = Join-Path $HOME "dataset_explorer\data\rir_meta_v3.db"
$env:RIR_DB_PATH = $env:RIR_DB
$env:RIR_DATA_ROOT = "D:\RIR_Data"
$env:RIR_PATH_MAP = "/old/workspace/data/=D:\RIR_Data\"
$env:RIR_ALLOW_OUTSIDE = "0"
$env:TMPDIR = Join-Path $HOME "RIR_Tmp"
$env:RIR_EXPORTS = Join-Path $HOME "RIR_Exports"
$env:RIR_COPY_ROOT = $env:TMPDIR
New-Item -ItemType Directory -Force $env:TMPDIR, $env:RIR_EXPORTS | Out-Null
Test-Path $env:RIR_DB
python -m streamlit run app\app.py --server.address 127.0.0.1 --server.port 8501
~~~

Update the example old root and `D:\RIR_Data` to match the source paths in your SQLite file and the actual destination. Without the recordings, the app still supports **metadata exploration and CSV**; it cannot reconstruct the missing audio.

**Using a server-hosted app from another computer:** Keep the datasets and SQLite database on the server, run Streamlit there, and connect to its interface using a private, authenticated connection approved by the host institution. You do not need a second copy of the data or Streamlit on the viewer's laptop. A localhost URL refers to the machine on which it is opened. Do not expose private research datasets through an unrestricted public tunnel.

### 9.2 Institutional Linux installation — maintainer's existing paths

The following paths document the installation used during project development. They are **specific to that system**, not universal requirements. They do not grant access; use an appropriately authorized account. No passwords, access keys or internal network endpoints are provided.

| Component | Configured location |
| --- | --- |
| Repository checkout | `/home/on61ewex/code/dataset_explorer/` |
| SQLite metadata | `/home/on61ewex/Database/db/rir_meta_v3.db` |
| Original RIR files | `/home/on61ewex/Database/data/` |
| Default host copy root | `/home/on61ewex/Database/tmp/` |
| Optional ZIP copies | `/home/on61ewex/Database/data/_exports/` |

**Run on the Linux host itself:**

~~~bash
cd /home/on61ewex/code/dataset_explorer
source .venv/bin/activate
export RIR_DB="/home/on61ewex/Database/db/rir_meta_v3.db"
export RIR_DB_PATH="$RIR_DB"
export RIR_DATA_ROOT="/home/on61ewex/Database/data"
export TMPDIR="/home/on61ewex/Database/tmp"
export RIR_EXPORTS="/home/on61ewex/Database/data/_exports"
export RIR_COPY_ROOT="$TMPDIR"
export RIR_ALLOW_OUTSIDE=0
unset RIR_PATH_MAP
mkdir -p "$TMPDIR" "$RIR_EXPORTS"

# Verify that the database is populated and the recording tree is available.
test -r "$RIR_DB" && echo "Database readable"
sqlite3 "$RIR_DB" "SELECT COUNT(*) FROM rirs;"
test -d "$RIR_DATA_ROOT" && echo "Data directory exists"
find "$RIR_DATA_ROOT" -type f \( -iname '*.wav' -o -iname '*.sofa' -o -iname '*.mat' \) -print -quit

python -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501
~~~

With these settings, **all SQL queries use the Linux SQLite file**, and **the local ZIP builder looks for recordings under the Linux data tree**. Search results alone do not prove all recording paths resolve. In the interface, select a small dataset, click **Search**, inspect **Path check (first 25)**, then build a small ZIP. Verify that the copied-file count is positive and inspect `zip_diagnostics.txt` for any missing files.

A user who views this Linux-hosted Streamlit instance through an approved private connection can download the selected ZIP through their own browser; the raw corpus stays on the Linux host.

### 9.3 Environment configuration

| Variable | Meaning |
| --- | --- |
| `RIR_DB` | SQLite filename |
| `RIR_DB_PATH` | Optional higher-priority SQLite override |
| `RIR_DATA_ROOT` | Root directory for local recording lookup |
| `RIR_PATH_MAP` | Optional semicolon-separated `old-prefix=new-prefix` mappings |
| `RIR_ALLOW_OUTSIDE` | `0` (default) restricts ZIP copying to `RIR_DATA_ROOT`; `1` permits other readable paths |
| `TMPDIR` | Writable temporary folder; also the default base for host copies |
| `RIR_EXPORTS` | Optional saved ZIP archives on the Streamlit host |
| `RIR_COPY_ROOT` | Authorized parent directory for the no-compression copy operation; destination must stay within this root |
| `PUBLIC_DATA_BASE_URL` | Optional externally hosted URL prefix; it does **not** upload data |

The `file_path` column stores file-system locations. A corresponding `public_url` is meaningful only when an external file server has separately been configured and authorized.

### 9.4 Validation checklist and troubleshooting

- [ ] SQLite passes `PRAGMA integrity_check`, contains `rirs`, and has the expected row count.
- [ ] At least one representative `file_path` resolves to a readable recording on the machine running Streamlit.
- [ ] Installation succeeds in a fresh Python 3.11 virtual environment.
- [ ] Search, distribution plots and CSV export work.
- [ ] A small ZIP contains audio in `files/`, plus `manifest.csv` and `zip_diagnostics.txt`.
- [ ] A no-compression copy produces a unique `rir_copy_*` folder under `RIR_COPY_ROOT` with original recordings and manifest, without modifying the source files.
- [ ] T60/C50/C80 results are checked against representative reference signals before scientific use.
- [ ] Dataset licenses, institutional permissions and disk quotas permit the intended copying and sharing.

| Symptom | Likely cause or next check |
| --- | --- |
| `no such table: rirs` | Wrong or empty database; follow §4 |
| `unable to open database file` | Missing path or insufficient permissions |
| `streamlit` not found | Activate the environment; try `python -m streamlit` |
| Browser cannot open `0.0.0.0` | Use the actual hostname or `127.0.0.1` on the host |
| ZIP contains only metadata/diagnostics | Source recordings missing, path mapping incorrect, or outside-root policy |
| Export stalls | Limit the result set; ZIP creation compresses directly from original files but holds the completed download in memory |
| New datasets not visible | Click Search; cached dataset options may take up to five minutes to refresh |


## 10. Research reproducibility and responsible data use

For a publication or dataset release, record:

1. Dataset names, original authors, URLs, release versions, licenses, and access dates.
2. SHA-256 digests of original archives and SQLite snapshots; do not silently change source audio.
3. The Git commit hash (`git rev-parse HEAD`), Python version, dependency versions (`python -m pip freeze`), and exact commands used.
4. Calibration assumptions, response/channel selection, fit intervals, invalid-signal handling, missing-value counts, and any dataset-specific metadata parsing.
5. Whether fields represent direct measurements, estimated parameters, or proxies; keep `t60_method` and `bp_source` alongside reported values.

Quick quality checks:

~~~bash
sqlite3 "$RIR_DB" "PRAGMA integrity_check;"
sqlite3 "$RIR_DB" "SELECT dataset, COUNT(*), SUM(t60_s IS NOT NULL), SUM(c50_db IS NOT NULL), SUM(c80_db IS NOT NULL) FROM rirs GROUP BY dataset;"
sha256sum "$RIR_DB"   # Linux/WSL
git rev-parse HEAD
python --version
~~~

**Citing this software:** cite the repository URL and the exact commit/release you used, as well as the original RIR datasets and any standards/algorithms relevant to your analysis. See `CITATION.cff` for software citation metadata. No DOI, formal publication, software license, or research affiliation is asserted here without a corresponding record. Copyright and redistribution conditions of third-party recordings remain with their owners.

## 11. Project maintenance

- Edit scripts in a feature branch; test against a copy of the database.
- Verify `python -m py_compile app/app.py app/export_utils.py tools/ingest_bras_audio.py scripts/backfill_t60.py scripts/backfill_bp.py`.
- Run offline file-export regression tests: `python -m unittest discover -s tests -v`.
- Keep generated databases and raw archives out of commits. Review `git status` before pushing.
- The dataset-specific ingester and acoustic loaders should be extended and validated when introducing a new file format or corpus structure.

**Maintainer:** [KhashayarModares](https://github.com/KhashayarModares). Issues and reproducible bug reports: [repository issues](https://github.com/KhashayarModares/dataset_explorer/issues).
