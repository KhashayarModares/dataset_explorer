
```markdown
# 🎧 RIR Explorer — Room Impulse Response Metadata App

RIR Explorer is an interactive web application for exploring, visualizing, and exporting metadata from Room Impulse Response (RIR) datasets.

It is built entirely in Python using:

- **Streamlit** → creates the interactive web user interface.
- **SQLAlchemy** → acts as the “translator” that lets Python talk to the database.
- **SQLite** → stores all RIR metadata locally in a fast, portable `.db` file.

Together, they create a self-contained, browser-based app for dataset filtering, analysis, and metadata export.

---

## 🧠 How It Works — Intuitive Overview

| Layer | Role | What it does |
|-------|------|--------------|
| **Streamlit (`app/app.py`)** | User Interface | Displays data filters, charts, and export buttons in the browser. |
| **SQLAlchemy** | Translator | Connects the Streamlit app and Python scripts to the SQLite database. |
| **SQLite (`rir_meta_v3.db`)** | Database | Stores all RIR metadata (dataset info, room types, formats, T60, etc.). |
| **`ingest_bras_audio.py`** | Dataset Ingestion | Scans new datasets, extracts metadata (format, sample rate, duration, etc.), and stores them in the database. |
| **`backfill_t60.py`** | Acoustic Analysis | Computes missing acoustic parameters such as T60, C50, and C80 for existing database entries. |

---

## 📁 Project Folder Structure

Your directory layout should look like this:

```

/home/username/Database/
│
├── db/
│   └── rir_meta_v3.db          ← SQLite database (main metadata store)
│
├── data/                       ← Store all downloaded datasets here
│   ├── ADREAM/
│   ├── AaltoTransitions/
│   ├── ArniMcKenzie/
│   ├── ArniPrawda/
│   └── MyNewDataset/           ← You will add new datasets here
│
├── scripts/
│   ├── ingest_bras_audio.py    ← Ingests metadata for new datasets
│   └── backfill_t60.py         ← Computes T60, C50, and C80
│
└── app/
└── app.py                  ← Streamlit web app interface

````

---

## ⚙️ Setting Up Your Environment

### 1️⃣ Activate your Python virtual environment

```bash
cd ~/code/dataset_explorer
source .venv/bin/activate
````

### 2️⃣ Set environment variables

Update the paths if your database and data folder are located elsewhere.

```bash
export RIR_DB="/home/username/Database/db/rir_meta_v3.db"
export RIR_DATA_ROOT="/home/username/Database/data"
export TMPDIR="/home/username/Database/tmp"
export RIR_ALLOW_OUTSIDE=1
```

---

## 🚀 Running the Streamlit App

Run the Streamlit dashboard locally:

```bash
streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501
```

Then open in your browser:

👉 [http://127.0.0.1:8501](http://127.0.0.1:8501)

The UI allows you to:

* Filter by dataset, file format, sample rate, LOS/NLOS, RT60 range, etc.
* View interactive distributions (e.g., histograms for T60, volume).
* Download filtered results as `.csv` or `.zip`.

---

## 🧩 Adding a New Dataset

When you download a new RIR dataset, follow these steps to include it in the app.

### Step 1 — Place your dataset

Put your dataset under:

```
/home/username/Database/data/MyNewDataset/
```

For example:

```
/home/username/Database/data/MyNewDataset/
 ├── room1.wav
 ├── room2.wav
 └── notes.txt
```

---

### Step 2 — Ingest the new dataset

Run the ingestion script to read all files, extract metadata, and populate the database.

```bash
cd /home/username/Database/scripts
python ingest_bras_audio.py --dataset MyNewDataset
```

This script automatically:

* Scans all audio files in `MyNewDataset`.
* Reads attributes such as file name, format, number of channels, sample rate, and duration.
* Adds them into the SQLite table `rirs`.

You can confirm the new entries were added:

```bash
sqlite3 /home/username/Database/db/rir_meta_v3.db "SELECT dataset, COUNT(*) FROM rirs GROUP BY dataset;"
```

---

### Step 3 — Compute T60, C50, and C80 for your dataset

Once metadata is ingested, compute the acoustic metrics with the backfill script:

```bash
cd /home/username/Database/scripts
python backfill_t60.py --dataset MyNewDataset --method energy_decay
```

This script:

* Loads each `.wav` or `.sofa` file.
* Estimates the **reverberation time (T60)** and **clarity indices (C50, C80)**.
* Writes the results back into the same database rows.

---

### Step 4 — Explore in the App

Now open the Streamlit app again:

```bash
cd ~/code/dataset_explorer
streamlit run app/app.py
```

Your new dataset will appear in the sidebar under “Datasets.”
You can now visualize, filter, and export its metadata interactively.

---

## 🧮 Database Schema Overview

| Column           | Description                   |
| ---------------- | ----------------------------- |
| `dataset`        | Dataset folder name           |
| `file_name`      | File name of the audio        |
| `file_format`    | Format (.wav, .sofa, etc.)    |
| `sample_rate_hz` | Sampling rate (Hz)            |
| `num_channels`   | Number of audio channels      |
| `duration_s`     | Duration in seconds           |
| `room_type`      | Room classification           |
| `distance_m`     | Mic–source distance (m)       |
| `t60_s`          | Reverberation time (s)        |
| `c50_db`         | Clarity index C50 (dB)        |
| `c80_db`         | Clarity index C80 (dB)        |
| `is_binaural`    | Whether the RIR is binaural   |
| `los`            | Line-of-sight condition       |
| `file_path`      | Absolute file path in `data/` |

---

## 🧠 Example Workflow Summary

Here’s a complete sequence for adding and analyzing a new dataset:

```bash
cd ~/code/dataset_explorer
source .venv/bin/activate
export RIR_DB="/home/username/Database/db/rir_meta_v3.db"
export RIR_DATA_ROOT="/home/username/Database/data"
export TMPDIR="/home/username/Database/tmp"
export RIR_ALLOW_OUTSIDE=1

# Copy your new dataset into the data folder
cp -r /path/to/MyNewDataset /home/username/Database/data/

# Ingest metadata into the database
python /home/username/Database/scripts/ingest_bras_audio.py --dataset MyNewDataset

# Compute T60, C50, and C80
python /home/username/Database/scripts/backfill_t60.py --dataset MyNewDataset

# Launch the interactive dashboard
streamlit run app/app.py
```

---

## 💡 Understanding Streamlit + SQLAlchemy + SQLite

Think of this app as a **three-layer system**:

1. **SQLite** → stores all metadata locally (`rir_meta_v3.db`).
2. **SQLAlchemy** → translates between your Python code and SQL queries (you don’t need to write SQL manually).
3. **Streamlit** → runs a web interface where all data is shown, filtered, and visualized interactively.

When you click “Search” in the app:

* Streamlit collects your filter inputs.
* SQLAlchemy dynamically builds and runs SQL queries on the SQLite database.
* Results are displayed instantly as tables and charts.

---

## 🌍 (Optional) Share the App Publicly

If you want others to see your local app (without deploying it permanently):

```bash
~/.local/bin/cloudflared tunnel --protocol http2 --url http://127.0.0.1:8501
```

This will generate a temporary public URL like:

```
https://your-dataset-app.trycloudflare.com
```

---

## 🧰 Troubleshooting Tips

* If the app shows *“unable to open database file”*, check that:

  * `RIR_DB` points to the correct `.db` path.
  * You have read/write permissions on that file.
* Always activate your virtual environment before running Python scripts.
* Run ingestion scripts from the same Python environment that has SQLAlchemy and pandas installed.

---

## ⚡ Quick Start (for advanced users)

```bash
cd ~/code/dataset_explorer
source .venv/bin/activate
export RIR_DB="$HOME/Database/db/rir_meta_v3.db"
export RIR_DATA_ROOT="$HOME/Database/data"
python $RIR_DATA_ROOT/../scripts/ingest_bras_audio.py --dataset MyNewDataset
python $RIR_DATA_ROOT/../scripts/backfill_t60.py --dataset MyNewDataset
streamlit run app/app.py
```

---

## 📜 License

MIT License – free to use, modify, and share.



```
