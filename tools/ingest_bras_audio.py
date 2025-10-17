#!/usr/bin/env python3
import os, sys, glob, math, sqlite3, argparse, wave, re
from pathlib import Path

# ---- External manifest mode (imports & globals) ----
import csv, json
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Iterable, Dict, Any, Optional, List

# Postgres (Neon) via SQLAlchemy
from sqlalchemy import create_engine, text

# Environment configuration
PGURL = os.environ.get("RIR_PGURL")  # e.g., postgresql://...neon.tech/...?
RIR_MANIFEST = os.environ.get("RIR_MANIFEST")  # path or URL to CSV/TSV/JSON/NDJSON
RIR_MANIFEST_FORMAT = os.environ.get("RIR_MANIFEST_FORMAT", "auto")  # csv|tsv|json|ndjson|auto
# ----------------------------------------------------

import numpy as np
from datetime import datetime, timezone

# Optional loaders
try:
    import soundfile as sf        # robust WAV read
except Exception:
    sf = None
try:
    import scipy.io as sio        # MAT v5
except Exception:
    sio = None
try:
    import h5py                   # MAT v7.3 (HDF5)
except Exception:
    h5py = None
try:
    from netCDF4 import Dataset   # SOFA
except Exception:
    Dataset = None
try:
    import pandas as pd           # dEchorate CSV helper (kept)
except Exception:
    pd = None

# ---------------- Schema ----------------
COLUMNS = [
  ("dataset","TEXT"), ("file_path","TEXT"), ("file_name","TEXT"), ("file_format","TEXT"),
  ("sample_rate_hz","REAL"), ("num_channels","INTEGER"), ("num_receivers","INTEGER"),
  ("duration_s","REAL"), ("is_binaural","INTEGER"),
  ("room_type","TEXT"), ("transition","TEXT"), ("los","TEXT"), ("distance_m","REAL"),
  ("array_model","TEXT"), ("data_kind","TEXT"), ("ambisonics_order","INTEGER"),
  ("acoustic_config","TEXT"), ("source_id","TEXT"), ("receiver_id","TEXT"),
  ("rx_x","REAL"), ("rx_y","REAL"), ("rx_z","REAL"),
  ("src_x","REAL"), ("src_y","REAL"), ("src_z","REAL"),
  ("t60_s","REAL"), ("t60_method","TEXT"),
  ("room_name","TEXT"),
  # Extra descriptive / hierarchical metadata
  ("position_label","TEXT"),
  ("microphone_model","TEXT"),
  ("source_model","TEXT"),
  ("coordinate_system","TEXT"),
  ("room_dims_m","TEXT"),
  ("original_relpath","TEXT"),
  ("inserted_at","TEXT")
]

def utc(): return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")

def migrate(con):
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rirs'")
    if not cur.fetchone():
        cur.execute("CREATE TABLE rirs (id INTEGER PRIMARY KEY AUTOINCREMENT, dataset TEXT, file_path TEXT)")
    have = {r[1] for r in cur.execute("PRAGMA table_info(rirs)")}
    for col,typ in COLUMNS:
        if col not in have:
            cur.execute(f"ALTER TABLE rirs ADD COLUMN {col} {typ}")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rirs_dataset ON rirs(dataset)")
    try:
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uidx_rirs_file_path ON rirs(file_path)")
    except sqlite3.OperationalError:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rirs_file_path ON rirs(file_path)")
    con.commit()

# ---------------- BUT ReverbDB helpers (harmless if not BUT layout) ----------------
KV_RE = re.compile(r'^\s*\$(\S+)\s+(.*\S)\s*$')
def read_meta_txt(path: Path) -> dict:
    d = {}
    if not path or not path.exists(): return d
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = KV_RE.match(line);  
                if not m: continue
                k, v = m.group(1), m.group(2)
                try: vv = float(v)
                except ValueError: vv = v.strip()
                d[k] = vv
    except Exception:
        pass
    return d

def _pick(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def parse_but_meta_for_wav(wav_path: Path):
    try:
        dataset_dir  = wav_path.parent.name
        mic_id_dir   = wav_path.parent.parent
        spk_setup    = mic_id_dir.parent
        mic_setup    = spk_setup.parent
        place_dir    = mic_setup.parent

        mic_meta = mic_id_dir / "mic_meta.txt"
        spk_meta = spk_setup / "spk_meta.txt"
        env_meta = place_dir / "env_meta.txt"
        env_full = place_dir / "env_full_meta.txt"

        dm = {}
        dm.update(read_meta_txt(env_full))
        dm.update(read_meta_txt(env_meta))
        dm.update(read_meta_txt(spk_meta))
        dm.update(read_meta_txt(mic_meta))

        room_name = _pick(dm, "EnvName") or place_dir.name
        room_type = _pick(dm, "EnvType")
        los       = _pick(dm, "EnvMic1SpeakerVisibility", "EnvMicSpeakerVisibility")

        mic_id = None
        try: mic_id = int(_pick(dm, "EnvMicID"))
        except Exception: mic_id = None
        if mic_id is not None:
            los = _pick(dm, f"EnvMic{mic_id}SpeakerVisibility","EnvMicSpeakerVisibility","EnvMic1SpeakerVisibility") or los

        src_depth = _pick(dm, "EnvSpk1Depth");  src_width = _pick(dm, "EnvSpk1Width");  src_height = _pick(dm, "EnvSpk1Height")
        keypref   = f"EnvMic{mic_id}" if mic_id is not None else "EnvMic1"
        rx_depth  = _pick(dm, f"{keypref}Depth", "EnvMicDepth", "EnvMic1Depth")
        rx_width  = _pick(dm, f"{keypref}Width", "EnvMicWidth", "EnvMic1Width")
        rx_height = _pick(dm, f"{keypref}Height", "EnvMicHeight", "EnvMic1Height")

        distance  = _pick(dm, f"{keypref}RelDistance","EnvMicRelDistance",
                             f"{keypref}RelDistanceRIRMeasured","EnvMicRelDistanceRIRMeasured",
                             f"{keypref}Distance","EnvMicDistance")

        D  = _pick(dm, "EnvDepth");  W = _pick(dm, "EnvWidth");  H = _pick(dm, "EnvHeight")
        D2 = _pick(dm, "Env2Depth"); W2= _pick(dm, "Env2Width"); H2= _pick(dm, "Env2Height")
        dims_txt = None
        if D and W and H:
            dims_txt = f"DWH={D}x{W}x{H}m"
            if D2 and W2 and H2:
                dims_txt += f"; L2={D2}x{W2}x{H2}m"

        mic_count = 0
        try:
            for child in spk_setup.iterdir():
                if child.is_dir() and (child / "mic_meta.txt").exists():
                    mic_count += 1
        except Exception:
            pass
        array_model = None
        mic_setup_name = _pick(dm, "EnvMicSetupName")
        mic_setup_id   = _pick(dm, "EnvMicSetupID")
        if mic_setup_name or mic_setup_id or mic_count:
            array_model = f"MicSetupName={mic_setup_name};MicSetupID={mic_setup_id};mics_total={mic_count}"

        acoustic_config = "/".join([place_dir.name, mic_setup.name, spk_setup.name, mic_id_dir.name, dataset_dir])

        kind = dataset_dir.lower()
        if kind == "rir": data_kind = "rir"
        elif kind == "silence": data_kind = "noise"
        elif "english" in kind or "librispeech" in kind: data_kind = "speech_retx"
        else: data_kind = kind

        def _f(v): 
            try: return float(v)
            except Exception: return None

        rx_x = _f(rx_depth); rx_y = _f(rx_width); rx_z = _f(rx_height)
        src_x= _f(src_depth);src_y= _f(src_width);src_z= _f(src_height)
        try:
            dist = float(distance) if distance not in (None, "", "None") else None
        except Exception:
            dist = None

        return dict(room_name=room_name, room_type=room_type, los=los,
                    rx_x=rx_x, rx_y=rx_y, rx_z=rx_z,
                    src_x=src_x, src_y=src_y, src_z=src_z,
                    distance_m=dist, acoustic_config=dims_txt,
                    array_model=array_model, data_kind=data_kind)
    except Exception:
        return {}

# ---------------- Generic readers ----------------
def wav_extract_meta(path: str):
    """Robust WAV header read. Returns (sr,ch,n_receivers,duration,is_binaural)."""
    sr = ch = nframes = None
    if sf is not None:
        info = sf.info(path)
        sr = info.samplerate; ch = info.channels; nframes = info.frames
    else:
        with wave.open(path, "rb") as w:
            sr = w.getframerate(); ch = w.getnchannels(); nframes = w.getnframes()
    dur = (nframes/float(sr)) if (nframes and sr) else None
    is_binaural = 1 if ch == 2 else 0 if ch is not None else None
    return sr, ch, 1, dur, is_binaural

def sofa_extract_meta(path):
    if Dataset is None:
        raise RuntimeError("netCDF4 not installed for SOFA support")
    with Dataset(path, "r") as nc:
        sr = None
        for k in ("Data.SamplingRate","SamplingRate","Data_SamplingRate","Fs"):
            if k in nc.variables:
                try: sr = float(np.squeeze(nc.variables[k][:])); break
                except Exception: pass
        if sr is None:
            for k in ("SamplingRate","Data_SamplingRate","Fs"):
                if hasattr(nc, k):
                    try: sr = float(np.squeeze(getattr(nc, k))); break
                    except Exception: pass
        ir = nc.variables.get("Data.IR")
        n_time = None; n_rx = None
        if ir is not None and getattr(ir, "ndim", 0) >= 3:
            n_time = ir.shape[-1]; n_rx = ir.shape[-2]
        elif ir is not None and getattr(ir, "ndim", 0) == 2:
            n_rx, n_time = ir.shape
        elif ir is not None and getattr(ir, "ndim", 0) == 1:
            n_time = ir.shape[0]; n_rx = 1
        n_ch = n_rx
        dur = (n_time / sr) if (sr and n_time) else None

        def to_cart(vec, units=None, ctype=None):
            a = np.array(vec, dtype=float).reshape(-1)
            if a.size < 3: return (None,None,None)
            if (ctype or "").lower().startswith("cart"):
                if units and isinstance(units, str) and units.lower().startswith("cm"):
                    a = a * 0.01
                return (float(a[0]), float(a[1]), float(a[2]))
            az,el,r = a[0], a[1], a[2]
            if units and isinstance(units, str) and units.lower().startswith("cm"):
                r *= 0.01
            azr, elr = math.radians(az), math.radians(el)
            x = r * math.cos(elr) * math.cos(azr)
            y = r * math.cos(elr) * math.sin(azr)
            z = r * math.sin(elr)
            return (float(x), float(y), float(z))

        rx_x=rx_y=rx_z=src_x=src_y=src_z=None
        if "ReceiverPosition" in nc.variables:
            rp = nc.variables["ReceiverPosition"]
            try:
                vec = rp[0, :3, 0] if rp.ndim>=3 else rp[:3]
                rx_x,rx_y,rx_z = to_cart(vec, getattr(rp,"Units",None), getattr(rp,"Type",None))
            except Exception: pass
        if "SourcePosition" in nc.variables:
            sp = nc.variables["SourcePosition"]
            try:
                vec = sp[0, :3, 0] if sp.ndim>=3 else sp[:3]
                src_x,src_y,src_z = to_cart(vec, getattr(sp,"Units",None), getattr(sp,"Type",None))
            except Exception: pass

        dist=None
        if None not in (rx_x,rx_y,rx_z,src_x,src_y,src_z):
            dx,dy,dz = rx_x-src_x, rx_y-src_y, rx_z-src_z
            dist = float((dx*dx + dy*dy + dz*dz) ** 0.5)

        is_binaural = 1 if (n_ch == 2) else 0 if n_ch is not None else None
        return sr, n_ch, n_rx, dur, is_binaural, (rx_x,rx_y,rx_z), (src_x,src_y,src_z), dist

def _as_numpy(v):
    try:
        import h5py as _h
        if isinstance(v, _h.Dataset): return np.array(v[()])
    except Exception: pass
    return np.array(v)

def _search_keys(obj, keys, depth=3):
    if depth < 0 or obj is None: return None
    if hasattr(obj, "keys"):
        try: klower = {str(k).lower(): k for k in obj.keys()}
        except Exception: klower = {}
        for want in keys:
            k = klower.get(want.lower())
            if k is not None:
                try: return _as_numpy(obj[k])
                except Exception: pass
        for k in list(getattr(obj, "keys", lambda: [])()):
            if str(k).startswith("__"): continue
            try: child = obj[k]
            except Exception: continue
            r = _search_keys(child, keys, depth-1)
            if r is not None: return r
    if isinstance(obj, np.void) and getattr(obj, "dtype", None) is not None and obj.dtype.names:
        names = [n for n in obj.dtype.names]
        lower = {n.lower(): n for n in names}
        for want in keys:
            name = lower.get(want.lower())
            if name is not None:
                try: return _as_numpy(getattr(obj, name))
                except Exception: pass
        for name in names:
            try: r = _search_keys(getattr(obj, name), keys, depth-1)
            except Exception: r=None
            if r is not None: return r
    if isinstance(obj, (list, tuple, np.ndarray)):
        it = obj
        try: it = obj.flat
        except Exception: pass
        for i,v in enumerate(it):
            r = _search_keys(v, keys, depth-1)
            if r is not None: return r
            if i>50: break
    return None

def load_mat_any(path):
    if sio is not None:
        try: return sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        except NotImplementedError: pass
        except Exception: pass
    if h5py is not None:
        try: return h5py.File(path, "r")
        except Exception: pass
    raise RuntimeError("Cannot open MAT (need scipy.io or h5py)")

def mat_extract_meta(path):
    obj = load_mat_any(path)
    sr = None
    sr_arr = _search_keys(obj, ["fs","sampling_rate","sr","Fs","FS"])
    if sr_arr is not None:
        try: sr = float(np.squeeze(sr_arr))
        except Exception: sr = None
    ir = _search_keys(obj, ["h","ir","brir","rir","impulse_response"])
    n_time = n_ch = None
    if ir is not None:
        arr = np.array(ir)
        if arr.ndim == 1:
            n_time = arr.shape[0]; n_ch = 1
        else:
            tdim = int(np.argmax(arr.shape))
            n_time = int(arr.shape[tdim])
            other = int(np.prod([arr.shape[i] for i in range(arr.ndim) if i != tdim]))
            n_ch = other if other > 0 else 1
    dur = (n_time / sr) if (sr and n_time) else None
    is_binaural = 1 if (n_ch == 2) else 0 if n_ch is not None else None
    return sr, n_ch, n_ch, dur, is_binaural, (None,None,None), (None,None,None), None

# ---------------- Huddersfield 360° BRIR enrichment ----------------
POS_DIST_RE = re.compile(r'^(LW|L|C)\s*([0-9]+)\s*m\b', re.IGNORECASE)

def huddersfield_enrich(path: Path, rec: dict, root: Path):
    low_full = str(path).lower()
    name = path.name
    parent_names = [p.name.lower() for p in path.parents]

    m = POS_DIST_RE.match(name)
    if m:
        plabel = m.group(1).upper()
        try: dist = float(m.group(2))
        except Exception: dist = None
        rec.setdefault("position_label", plabel)
        if rec.get("distance_m") is None and dist is not None:
            rec["distance_m"] = dist

    is_foa = "/foa/" in low_full
    is_binaural = "/binaural/" in low_full or name.lower().endswith(".sofa")
    is_omni = ("/omni" in low_full) or ("/omnidirectional" in low_full)

    if is_foa:
        rec.setdefault("room_type","concert hall")
        rec.setdefault("room_name","Huddersfield_Concert_Hall")
        rec.setdefault("microphone_model","Sennheiser Ambeo (FOA)")
        rec["ambisonics_order"] = rec.get("ambisonics_order") or 1
        if rec.get("is_binaural") is None: rec["is_binaural"] = 0
        if any("a format" in p for p in parent_names):
            rec.setdefault("data_kind","foa_a"); rec.setdefault("array_model","Ambeo A-format")
        if any("b format" in p for p in parent_names) or "ambix" in low_full:
            rec.setdefault("data_kind","foa_b"); rec.setdefault("array_model","Ambeo B-format (AmbiX)")

    if is_binaural:
        rec.setdefault("room_type","concert hall")
        rec.setdefault("room_name","Huddersfield_Concert_Hall")
        rec.setdefault("microphone_model","Neumann KU100")
        rec.setdefault("data_kind","brir")
        rec["is_binaural"] = 1

    if is_omni:
        rec.setdefault("room_type","concert hall")
        rec.setdefault("room_name","Huddersfield_Concert_Hall")
        rec.setdefault("microphone_model","DPA 4006 (omni)")
        rec.setdefault("data_kind","omni_rir")
        rec["is_binaural"] = 0
        rec["ambisonics_order"] = rec.get("ambisonics_order") or 0

    rec.setdefault("coordinate_system","stage-centric (dataset doc); units in meters")
    try:
        rec["original_relpath"] = str(path.relative_to(root))
    except Exception:
        rec["original_relpath"] = path.name
    return rec

# ---------------- Ilmenau A.LI.EN enrichment ----------------
ILM_CM_RE = re.compile(r'([0-9]+(?:\.[0-9]+)?)\s*cm\b', re.IGNORECASE)
ILM_M_RE  = re.compile(r'([0-9]+(?:\.[0-9]+)?)\s*m\b',  re.IGNORECASE)
ILM_DEG_RE= re.compile(r'(\d{1,3})\s*deg\b',           re.IGNORECASE)

def ilmenau_enrich(path: Path, rec: dict, root: Path):
    low = str(path).lower()
    name = path.name.lower()

    rec.setdefault("room_type", "living room")
    rec.setdefault("room_name", "Ilmenau_LivingRoom")
    rec.setdefault("room_dims_m", "DWH=4.54x3.27x3.46m")
    rec.setdefault("coordinate_system", "room plan; units in meters")

    if "kemar" in low:
        rec.setdefault("microphone_model", "KEMAR 45BA")
        rec.setdefault("data_kind", "brir")
        if rec.get("is_binaural") is None: rec["is_binaural"] = 1
    if "omni" in low or "omnidirectional" in low or "m2211" in low or "nti" in low:
        rec.setdefault("microphone_model", "NTi M2211 + MA220 (omni)")
        rec.setdefault("data_kind", "omni_rir")
        rec["is_binaural"] = 0
        rec["ambisonics_order"] = rec.get("ambisonics_order") or 0
    if "sdm" in low:
        rec.setdefault("microphone_model", "Ilmenau SDM (6x Primo EM258 + Earthworks M30)")
        rec.setdefault("data_kind", "sdm_rir")
        rec["is_binaural"] = 0
    if "eigenmike" in low:
        rec.setdefault("microphone_model", "Eigenmike")
        rec.setdefault("data_kind", "srir")
        rec["is_binaural"] = 0

    if rec.get("distance_m") is None:
        m = ILM_M_RE.search(name)
        if m:
            try: rec["distance_m"] = float(m.group(1))
            except Exception: pass
    if rec.get("distance_m") is None:
        m = ILM_CM_RE.search(name)
        if m:
            try: rec["distance_m"] = float(m.group(1)) / 100.0
            except Exception: pass

    if rec.get("distance_m") is None:
        for pid, d in (("p1",2.0),("p2",2.0),("p3",3.5),("p4",5.5),("p5",0.8)):
            if pid in name:
                rec["distance_m"] = d
                break

    ac = rec.get("acoustic_config")
    mdeg = ILM_DEG_RE.search(name)
    if mdeg:
        try:
            deg = int(mdeg.group(1))
            extra = f"orientation_deg={deg}"
            rec["acoustic_config"] = "; ".join([s for s in [ac, extra] if s])
        except Exception:
            pass

    try:
        rec["original_relpath"] = str(path.relative_to(root))
    except Exception:
        rec["original_relpath"] = path.name
    return rec

# ---------------- dEchorate helpers (kept; harmless if not present) ----------------
def _find_dechorate_assets(root: Path):
    h5 = None; csv = None
    for p in root.glob("*.hdf5"):
        if "dechorate" in p.name.lower() and "rir" in p.name.lower():
            h5 = p; break
    candidates = list(root.glob("*dEchorate*database*.csv")) + list(root.glob("*database*.csv")) + list(root.glob("*.csv"))
    csv = candidates[0] if candidates else None
    return h5, csv

def _col(df, *alts):
    for a in alts:
        if a in df.columns: return a
        for c in df.columns:
            if c.lower() == a.lower(): return c
    return None

def ingest_dechorate_from_csv(cur, dataset: str, root: Path):
    if pd is None:
        print("[warn] pandas not available; skipping dEchorate CSV ingest.")
        return 0, 0
    h5_path, csv_path = _find_dechorate_assets(root)
    if csv_path is None:
        print("[warn] dEchorate CSV not found; skipping CSV ingest.")
        return 0, 0
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    c_rirpath = _col(df, "rir_path","h5_path","dataset_path","path_in_h5","hdf5_key","filename")
    c_ririd   = _col(df, "rir_id","id","index")
    c_fs      = _col(df, "fs","sr","sampling_rate_hz","rir_fs","sample_rate","SampleRate","Fs","fs_hz")
    c_room    = _col(df, "room_name","room","env_name","room_code")
    c_cond    = _col(df, "condition","surface_state","state","config")
    c_array   = _col(df, "array_id","array","mic_array","mic_setup")
    c_mic     = _col(df, "mic_id","mic","receiver_id")
    c_src     = _col(df, "src_id","source_id","source")
    c_rx_x    = _col(df, "mic_pos_x","mic_x","rx_x","receiver_x","array_bar_pos_x")
    c_rx_y    = _col(df, "mic_pos_y","mic_y","rx_y","receiver_y","array_bar_pos_y")
    c_rx_z    = _col(df, "mic_pos_z","mic_z","rx_z","receiver_z","array_bar_pos_z")
    c_sx_x    = _col(df, "src_pos_x","src_x","source_x")
    c_sx_y    = _col(df, "src_pos_y","src_y","source_y")
    c_sx_z    = _col(df, "src_pos_z","src_z","source_z")
    c_dist    = _col(df, "distance","distance_m","mic_src_distance")
    c_rd      = _col(df, "room_depth","depth")
    c_rw      = _col(df, "room_width","width")
    c_rh      = _col(df, "room_height","height")

    n_ok = n_fail = 0
    default_room_type = "shoebox room"

    for _, r in df.iterrows():
        inner = None
        if c_rirpath and pd.notna(r.get(c_rirpath)):
            inner = str(r.get(c_rirpath))
        elif c_ririd and pd.notna(r.get(c_ririd)):
            inner = f"id={r.get(c_ririd)}"

        if h5_path is not None and inner:
            file_path = f"{str(h5_path.resolve())}::{inner}"
            file_name = inner.split("/")[-1]
            file_format = "HDF5"
        elif h5_path is not None:
            file_path = str(h5_path.resolve())
            file_name = h5_path.name
            file_format = "HDF5"
        else:
            file_path = str(root.resolve())
            file_name = Path(file_path).name
            file_format = "HDF5"

        fs = float(r.get(c_fs)) if (c_fs and pd.notna(r.get(c_fs))) else None
        rx_x = float(r.get(c_rx_x)) if (c_rx_x and pd.notna(r.get(c_rx_x))) else None
        rx_y = float(r.get(c_rx_y)) if (c_rx_y and pd.notna(r.get(c_rx_y))) else None
        rx_z = float(r.get(c_rx_z)) if (c_rx_z and pd.notna(r.get(c_rx_z))) else None
        src_x = float(r.get(c_sx_x)) if (c_sx_x and pd.notna(r.get(c_sx_x))) else None
        src_y = float(r.get(c_sx_y)) if (c_sx_y and pd.notna(r.get(c_sx_y))) else None
        src_z = float(r.get(c_sx_z)) if (c_sx_z and pd.notna(r.get(c_sx_z))) else None

        dist = float(r.get(c_dist)) if (c_dist and pd.notna(r.get(c_dist))) else None
        if dist is None and None not in (rx_x, rx_y, rx_z, src_x, src_y, src_z):
            dx, dy, dz = rx_x - src_x, rx_y - src_y, rx_z - src_z
            dist = float((dx*dx + dy*dy + dz*dz) ** 0.5)

        dims_txt = None
        if c_rd and c_rw and c_rh:
            try:
                D = float(r.get(c_rd)); W = float(r.get(c_rw)); H = float(r.get(c_rh))
                dims_txt = f"DWH={D}x{W}x{H}m"
            except Exception:
                pass

        rec = {
            "dataset": dataset,
            "file_path": file_path,
            "file_name": file_name,
            "file_format": file_format,
            "sample_rate_hz": fs,
            "num_channels": 1,
            "num_receivers": 1,
            "duration_s": None,
            "is_binaural": 0,
            "room_type": default_room_type,
            "transition": None,
            "los": None,
            "distance_m": dist,
            "array_model": str(r.get(c_array)) if c_array else None,
            "data_kind": "rir",
            "ambisonics_order": None,
            "acoustic_config": "; ".join([s for s in [
                (str(r.get(c_room)) if c_room else None),
                (str(r.get(c_cond)) if c_cond else None),
                (dims_txt if dims_txt else None)
            ] if s]),
            "source_id": str(r.get(c_src)) if c_src else None,
            "receiver_id": str(r.get(c_mic)) if c_mic else None,
            "rx_x": rx_x, "rx_y": rx_y, "rx_z": rx_z,
            "src_x": src_x, "src_y": src_y, "src_z": src_z,
            "t60_s": None, "t60_method": None,
            "room_name": str(r.get(c_room)) if c_room else "dEchorate_room",
            "position_label": None,
            "microphone_model": None,
            "source_model": None,
            "coordinate_system": None,
            "room_dims_m": dims_txt,
            "original_relpath": None,
            "inserted_at": utc()
        }
        try:
            upsert(cur, rec); n_ok += 1
        except Exception as e:
            n_fail += 1
            print("[dEchorate CSV] fail:", e)
    return n_ok, n_fail

# ---------------- Upsert ----------------
def upsert(cur, rec: dict):
    cols = [c for c,_ in COLUMNS]
    ph = ",".join(["?"]*len(cols))
    vals = [rec.get(c) for c in cols]
    try:
        sql = (
          f"INSERT INTO rirs ({','.join(cols)}) VALUES ({ph}) "
          "ON CONFLICT(file_path) DO UPDATE SET "
          "dataset=excluded.dataset, file_name=excluded.file_name, file_format=excluded.file_format, "
          "sample_rate_hz=excluded.sample_rate_hz, num_channels=excluded.num_channels, num_receivers=excluded.num_receivers, "
          "duration_s=excluded.duration_s, is_binaural=excluded.is_binaural, "
          "room_type=excluded.room_type, distance_m=excluded.distance_m, "
          "array_model=excluded.array_model, data_kind=excluded.data_kind, ambisonics_order=excluded.ambisonics_order, "
          "acoustic_config=excluded.acoustic_config, room_name=excluded.room_name, "
          "rx_x=excluded.rx_x, rx_y=excluded.rx_y, rx_z=excluded.rx_z, "
          "src_x=excluded.src_x, src_y=excluded.src_y, src_z=excluded.src_z, "
          "t60_s=excluded.t60_s, t60_method=excluded.t60_method, "
          "position_label=excluded.position_label, microphone_model=excluded.microphone_model, "
          "source_model=excluded.source_model, coordinate_system=excluded.coordinate_system, "
          "room_dims_m=excluded.room_dims_m, original_relpath=excluded.original_relpath, "
          "inserted_at=excluded.inserted_at"
        )
        cur.execute(sql, vals)
    except sqlite3.OperationalError:
        row = cur.execute("SELECT id FROM rirs WHERE file_path=?", (rec["file_path"],)).fetchone()
        if row:
            rid = row[0]
            cols_no_fp = [c for c,_ in COLUMNS if c!="file_path"]
            set_clause = ", ".join([f"{c}=?" for c in cols_no_fp])
            params = [rec.get(c) for c in cols_no_fp] + [rid]
            cur.execute(f"UPDATE rirs SET {set_clause} WHERE id=?", params)
        else:
            cur.execute(f"INSERT INTO rirs ({','.join(cols)}) VALUES ({ph})", vals)

# ---------------- Ingest one ----------------
def ingest_one(cur, path: Path, dataset: str, root: Path):
    pstr = str(path.resolve()); low = pstr.lower()
    fname = path.name

    fmt = None
    sr = dur = None
    n_ch = n_rx = None
    is_bin = None
    rx_x=rx_y=rx_z=src_x=src_y=src_z=None
    dist=None
    array_model = None
    acoustic_config = None
    data_kind = None
    room_type = None
    room_name = None
    los = None
    ambi_order = None
    position_label = None
    microphone_model = None
    source_model = None
    coordinate_system = None
    room_dims_m = None

    if low.endswith(".wav"):
        fmt = "WAV"; data_kind = "wav"
        sr, n_ch, n_rx, dur, is_bin = wav_extract_meta(pstr)

        try:
            meta = parse_but_meta_for_wav(path) or {}
        except Exception:
            meta = {}
        for k in ("room_type","room_name","los","acoustic_config","array_model","data_kind","distance_m"):
            v = meta.get(k)
            if v is not None:
                if k=="distance_m": dist = v
                elif k=="data_kind": data_kind = v
                elif k=="acoustic_config": acoustic_config=v
                elif k=="array_model": array_model=v
                elif k=="room_type": room_type=v
                elif k=="room_name": room_name=v
                elif k=="los": los=v

    elif low.endswith(".sofa"):
        fmt = "SOFA"; data_kind = "sofa"
        sr, n_ch, n_rx, dur, is_bin, (rx_x,rx_y,rx_z), (src_x,src_y,src_z), dist = sofa_extract_meta(pstr)

    elif low.endswith(".mat"):
        fmt = "MAT"; data_kind = "mat"
        sr, n_ch, n_rx, dur, is_bin, (rx_x,rx_y,rx_z), (src_x,src_y,src_z), dist = mat_extract_meta(pstr)
    else:
        return

    rec_tmp = dict(
        distance_m=dist, data_kind=data_kind, array_model=array_model,
        room_type=room_type, room_name=room_name, is_binaural=is_bin,
        ambisonics_order=ambi_order, position_label=position_label,
        microphone_model=microphone_model, source_model=source_model,
        coordinate_system=coordinate_system, room_dims_m=room_dims_m
    )
    rec_tmp = huddersfield_enrich(path, rec_tmp, root)
    rec_tmp = ilmenau_enrich(path, rec_tmp, root)

    dist = rec_tmp.get("distance_m", dist)
    data_kind = rec_tmp.get("data_kind", data_kind)
    array_model = rec_tmp.get("array_model", array_model)
    room_type = rec_tmp.get("room_type", room_type)
    room_name = rec_tmp.get("room_name", room_name)
    is_bin = rec_tmp.get("is_binaural", is_bin)
    ambi_order = rec_tmp.get("ambisonics_order", ambi_order)
    position_label = rec_tmp.get("position_label", position_label)
    microphone_model = rec_tmp.get("microphone_model", microphone_model)
    source_model = rec_tmp.get("source_model", source_model)
    coordinate_system = rec_tmp.get("coordinate_system", coordinate_system)
    room_dims_m = rec_tmp.get("room_dims_m", room_dims_m)
    original_relpath = rec_tmp.get("original_relpath")

    rec = {
        "dataset": dataset,
        "file_path": pstr,
        "file_name": fname,
        "file_format": fmt,
        "sample_rate_hz": sr,
        "num_channels": n_ch,
        "num_receivers": n_rx,
        "duration_s": dur,
        "is_binaural": is_bin,
        "room_type": room_type,
        "transition": None,
        "los": los,
        "distance_m": dist,
        "array_model": array_model,
        "data_kind": data_kind,
        "ambisonics_order": ambi_order,
        "acoustic_config": acoustic_config,
        "source_id": None, "receiver_id": None,
        "rx_x": rx_x, "rx_y": rx_y, "rx_z": rx_z,
        "src_x": src_x, "src_y": src_y, "src_z": src_z,
        "t60_s": None, "t60_method": None,
        "room_name": room_name,
        "position_label": position_label,
        "microphone_model": microphone_model,
        "source_model": source_model,
        "coordinate_system": coordinate_system,
        "room_dims_m": room_dims_m,
        "original_relpath": original_relpath,
        "inserted_at": utc()
    }
    upsert(cur, rec)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Unified ingester: WAV/SOFA/MAT (+dEchorate CSV if present).")
    ap.add_argument("--db", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--glob", default="**/*.*")
    ap.add_argument("--max", type=int, default=0, help="limit number of files for debug")
    ap.add_argument("--dechorate-csv-only", action="store_true", help="Only ingest dEchorate via CSV (skip file scan).")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    migrate(conn)
    cur = conn.cursor()

    root = Path(args.root)

    if args.dechorate_csv_only:
        try:
            ok, fail = ingest_dechorate_from_csv(cur, args.dataset, root)
            conn.commit()
            print(f"[dEchorate CSV] OK={ok}, FAIL={fail}")
        except Exception as e:
            print("[dEchorate CSV] skipped/failed:", e)
        conn.close(); return

    files = glob.glob(os.path.join(args.root, args.glob), recursive=True)
    files = [f for f in files if f.lower().endswith((".wav",".sofa",".mat"))]
    files.sort()
    if args.max and len(files) > args.max:
        files = files[:args.max]
    total = len(files)
    n_ok = n_fail = 0
    print(f"[info] Found {total} files to ingest (WAV/SOFA/MAT).")
    for i,f in enumerate(files, 1):
        try:
            ingest_one(cur, Path(f), args.dataset, root); n_ok += 1
        except Exception as e:
            print("Failed:", f, e); n_fail += 1
        if i % 200 == 0 or i == total:
            conn.commit()
            print(f"[prog] {i}/{total} committed...")
    conn.commit(); conn.close()
    print("Ingested: OK={}, FAIL={}".format(n_ok, n_fail))

if __name__ == "__main__":
    main()
