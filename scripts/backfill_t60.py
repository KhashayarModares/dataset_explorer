#!/usr/bin/env python3
import argparse, os, sys, sqlite3, zipfile, tempfile
from pathlib import Path
import numpy as np

# WAV via soundfile
try:
    import soundfile as sf
except Exception:
    sf = None

# SOFA via netCDF4
try:
    from netCDF4 import Dataset
except Exception:
    Dataset = None

# MAT via scipy/h5py
try:
    import scipy.io as sio
except Exception:
    sio = None
try:
    import h5py
except Exception:
    h5py = None

STD_RATES = [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000, 192000]

# ---------------- DB migrate ----------------
def migrate(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(rirs)")}
    if "t60_s" not in cols:
        con.execute("ALTER TABLE rirs ADD COLUMN t60_s REAL")
    if "t60_method" not in cols:
        con.execute("ALTER TABLE rirs ADD COLUMN t60_method TEXT")
    con.commit()

# ---------------- WAV readers ----------------
def read_wav_from_fs(path):
    if sf is None:
        raise RuntimeError("soundfile not available")
    x, fs = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return float(fs), x.astype(np.float32)

def read_wav_from_zip(zip_path, member):
    with zipfile.ZipFile(zip_path, "r") as zf, zf.open(member, "r") as f:
        data = f.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(data); tmp.flush()
        return read_wav_from_fs(tmp.name)

# ---------------- SOFA readers ----------------
def _sr_from_nc(nc):
    import numpy as _np
    for key in ("SamplingRate","Data.SamplingRate","Data_SamplingRate","Fs"):
        if key in nc.variables:
            try: return float(_np.squeeze(nc.variables[key][:]))
            except Exception: pass
    for key in ("SamplingRate","Data.SamplingRate","Fs"):
        if hasattr(nc, key):
            try: return float(_np.squeeze(getattr(nc, key)))
            except Exception: pass
    return None

def _mono_ir_from_nc(nc, fast_first=True):
    v = (nc.variables.get("Data.IR") or nc.variables.get("DataIR") or
         nc.variables.get("IR") or nc.variables.get("Data"))
    if v is None:
        raise RuntimeError("No Data.IR")
    sel = v[0, ...] if (fast_first and v.ndim >= 2) else v[:]
    arr = np.array(sel, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0)

def read_sofa_from_fs(path):
    if Dataset is None:
        raise RuntimeError("netCDF4 not installed for SOFA support")
    with Dataset(path, "r") as nc:
        sr = _sr_from_nc(nc)
        if not sr or sr <= 0:
            raise RuntimeError("SamplingRate not found")
        x  = _mono_ir_from_nc(nc, fast_first=True)
    return float(sr), x.astype(np.float32)

def read_sofa_from_zip(zip_path, member):
    if Dataset is None:
        raise RuntimeError("netCDF4 not installed for SOFA support")
    with zipfile.ZipFile(zip_path, "r") as zf, zf.open(member, "r") as f:
        data = f.read()
    with tempfile.NamedTemporaryFile(suffix=".sofa", delete=True) as tmp:
        tmp.write(data); tmp.flush()
        return read_sofa_from_fs(tmp.name)

# ---------------- MAT helpers ----------------
def _is_scalar_num(v):
    try:
        a = np.array(v)
        return a.ndim == 0 and np.isfinite(a).all()
    except Exception:
        return False

def _best_rate_from_numbers(candidates):
    # pick exact standard rate if present; else nearest within tolerance
    cand = [float(c) for c in candidates if np.isfinite(c)]
    for r in STD_RATES:
        if any(abs(c - r) < 1e-6 for c in cand):
            return float(r)
    if cand:
        # nearest standard rate if within 1%
        n = cand[0]
        nearest = min(STD_RATES, key=lambda r: abs(r-n))
        if abs(nearest - n)/nearest <= 0.01:
            return float(nearest)
    return None

def _fs_guess_from_name(name_lower):
    # AIR convention: "binaural" ≈ 48k, "phone" ≈ 16k
    if "phone" in name_lower:
        return 16000.0
    if "binaural" in name_lower:
        return 48000.0
    return None

def _flatten_ir(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        return a
    # choose the longest dimension as time, average the rest
    time_axis = int(np.argmax(a.shape))
    a = np.moveaxis(a, time_axis, -1)
    return a.reshape(-1, a.shape[-1]).mean(axis=0).astype(np.float32)

def read_mat_from_fs(path):
    """
    Robust MAT reader for Ilmenau A.LI.EN:
      - Recurses dict, numpy.void (MAT structs), mat_struct, object arrays, lists/tuples.
      - Any numeric array with ANY dimension >= 128 is a candidate IR.
      - Picks the candidate that yields the longest 1D time series after flattening.
      - FS from many key names/attrs; default 48k if absent.
    """
    import numpy as _np

    FS_KEYS = ("fs","Fs","FS","sr","SR","SamplingRate","samplingrate","sampling_rate",
               "SampleRate","SampleRateHz","Fs_Hz","fs_hz","Sampling_Frequency",
               "samplingfrequency","sampleratehz")
    TAGS = ("brir","rir","ir","impulse","impulseresponse","signal","sig","y","x",
            "left","right","hl","hr","omni","kemar","sdm","data","values")

    def _is_num_array(v):
        try:
            a = _np.asarray(v)
            return a.dtype.kind in "fiu" and a.size > 0
        except Exception:
            return False

    def _has_time_len(a):
        a = _np.asarray(a)
        return max(a.shape) if a.ndim >= 1 else 0

    def _flatten_to_mono(arr):
        a = _np.asarray(arr)
        if a.ndim == 0:
            return None
        if a.ndim == 1:
            return a.astype(_np.float32)
        # Choose the longest axis as time; average the rest
        tdim = int(_np.argmax(a.shape))
        a = _np.moveaxis(a, tdim, -1)
        a = a.reshape(-1, a.shape[-1]).mean(axis=0)
        return a.astype(_np.float32)

    def _is_mat_struct(x):
        return hasattr(x, '_fieldnames') or (hasattr(x,'__dict__') and bool(getattr(x,'__dict__',{})))

    def _iter_fields(x):
        if isinstance(x, _np.void) and getattr(x,'dtype',None) is not None and x.dtype.names:
            for n in x.dtype.names:
                yield n, getattr(x, n)
        elif hasattr(x,'_fieldnames') and x._fieldnames:
            for n in x._fieldnames:
                yield n, getattr(x, n)
        elif hasattr(x,'__dict__'):
            for n,v in x.__dict__.items():
                if not str(n).startswith("__"):
                    yield n, v

    # Try SciPy (v5)
    mat = None
    try:
        import scipy.io as _sio
        mat = _sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        mat = None
    except Exception:
        mat = None

    primary = []   # (score, time_len, array)
    fallback = []  # (score, time_len, array)
    rates = []

    fs_keys_lower = {k.lower() for k in FS_KEYS}

    def _collect_rate(k, v):
        try:
            kl = (str(k) or "").lower()
            if kl in fs_keys_lower:
                a = _np.array(v)
                val = float(a.squeeze())
                if 6000 <= val <= 192000:
                    rates.append(val)
        except Exception:
            pass

    def _maybe_add_array(name, arr):
        if not _is_num_array(arr):
            return
        tlen = _has_time_len(arr)
        if tlen < 128:   # ignore tiny vectors/matrices
            return
        mono = _flatten_to_mono(arr)
        if mono is None or mono.size < 128:
            return
        score = 1 if any(tag in (name or "").lower() for tag in TAGS) else 0
        if score:
            primary.append((score, mono.size, mono))
        else:
            fallback.append((score, mono.size, mono))

    def _walk(obj, name="", depth=0, max_depth=10):
        if depth > max_depth:
            return
        # dict
        if isinstance(obj, dict):
            for k,v in obj.items():
                if str(k).startswith("__"): continue
                _collect_rate(k, v)
                _walk(v, f"{name}.{k}" if name else str(k), depth+1)
            return
        # struct / mat_struct
        if isinstance(obj, _np.void) or _is_mat_struct(obj):
            for k,v in _iter_fields(obj):
                _collect_rate(k, v)
                _maybe_add_array(k, v)
                _walk(v, f"{name}.{k}" if name else str(k), depth+1)
            return
        # list / tuple
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _walk(v, f"{name}[{i}]", depth+1)
            return
        # object array (cells)
        if isinstance(obj, _np.ndarray) and obj.dtype == object:
            for idx, v in _np.ndenumerate(obj):
                _walk(v, f"{name}{list(idx)}", depth+1)
            return
        # numeric array
        if _is_num_array(obj):
            _maybe_add_array(name, obj)
            return
        # scalar -> maybe fs
        _collect_rate(name, obj)

    if mat is not None:
        if 'rirs' in mat:
            _walk(mat['rirs'], "rirs", 0)
        _walk(mat, "", 0)

    # Try HDF5 (v7.3)
    if (not primary and not fallback) or not rates:
        try:
            import h5py
            with h5py.File(path, "r") as f:
                def _attr_rates(o):
                    for ak,av in o.attrs.items():
                        if str(ak).lower() in fs_keys_lower:
                            try:
                                val = float(_np.array(av).squeeze())
                                if 6000 <= val <= 192000:
                                    rates.append(val)
                            except Exception:
                                pass
                def _walk_h5(o, p=""):
                    _attr_rates(o)
                    for k in o.keys():
                        child = o[k]; full = f"{p}/{k}" if p else k
                        _attr_rates(child)
                        if isinstance(child, h5py.Dataset):
                            data = child[()]
                            _maybe_add_array(full, data)
                        elif isinstance(child, h5py.Group):
                            _walk_h5(child, full)
                _walk_h5(f, "")
        except Exception:
            pass

    pick = None
    if primary:
        pick = max(primary, key=lambda t: t[1])[2]
    elif fallback:
        pick = max(fallback, key=lambda t: t[1])[2]
    if pick is None:
        raise RuntimeError("No IR found in MAT")

    # decide fs
    def _best_rate(cand):
        STD = [8000,11025,16000,22050,24000,32000,44100,48000,96000,192000]
        c = [float(x) for x in cand if _np.isfinite(x)]
        for r in STD:
            if any(abs(x-r) < 1e-6 for x in c): return float(r)
        if c:
            n=c[0]; near=min(STD, key=lambda r: abs(r-n))
            if abs(near-n)/near <= 0.01: return float(near)
        return None

    fs = _best_rate(rates) if rates else None
    if fs is None: fs = 48000.0
    return float(fs), pick.astype(_np.float32)

# ---------------- ZIP->MAT/SOFA readers ----------------
def read_mat_from_zip(zip_path, member):
    with zipfile.ZipFile(zip_path, "r") as zf, zf.open(member, "r") as f:
        data = f.read()
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=True) as tmp:
        tmp.write(data); tmp.flush()
        return read_mat_from_fs(tmp.name)


# ---------------- dEchorate HDF5 reader ----------------
def _h5_attr_float(obj, keys):
    import numpy as _np
    for k in keys:
        try:
            if k in obj.attrs:
                v = float(_np.array(obj.attrs[k]).squeeze())
                if 6000 <= v <= 192000:
                    return v
        except Exception:
            pass
    return None

def _to_1d(a):
    import numpy as _np
    a = _np.asarray(a, dtype=_np.float32)
    if a.ndim == 1: return a
    tdim = int(_np.argmax(a.shape))
    a = _np.moveaxis(a, tdim, -1)
    return a.reshape(-1, a.shape[-1]).mean(axis=0).astype(_np.float32)

def read_dechorate_from_hdf5(h5_path, member, sr_hint=None):
    import h5py, numpy as _np, os
    # 0) sample rate from attrs -> hint -> 48k
    with h5py.File(h5_path, "r") as f:
        sr = _h5_attr_float(f, ("fs","Fs","sampling_rate","SamplingRate"))
        if sr is None:
            for gname in ("rirs","RIRs","rir","RIR"):
                g = f.get(gname)
                if g is not None:
                    sr = _h5_attr_float(g, ("fs","Fs","sampling_rate","SamplingRate"))
                    if sr is not None: break
        if sr is None and sr_hint:
            try: sr = float(sr_hint)
            except Exception: pass
        if sr is None: sr = 48000.0

        # Helper: deref a dataset object to 1-D IR
        def _ds_to_ir(ds):
            x = ds[()]
            return _to_1d(x)

        # 1) explicit member path
        if member and member in f:
            obj = f[member]
            if isinstance(obj, h5py.Dataset):
                return float(sr), _ds_to_ir(obj)
            if isinstance(obj, h5py.Group):
                for k in ("rir","RIR","ir","IR"):
                    if k in obj and isinstance(obj[k], h5py.Dataset):
                        return float(sr), _ds_to_ir(obj[k])
                for k in obj.keys():
                    if isinstance(obj[k], h5py.Dataset) and obj[k].dtype.kind in "fF":
                        return float(sr), _ds_to_ir(obj[k])
                raise KeyError(f"HDF5 member exists but no dataset inside: {member}")

        # 2) id=N -> try CSV mapping to filename; else packed datasets with rows
        if member and member.startswith("id="):
            try: rid = int(member.split("=",1)[1])
            except Exception: rid = None
            # try CSV map in same dir
            try:
                import pandas as _pd
                csv_path = os.path.join(os.path.dirname(h5_path), "dEchorate_database.csv")
                if os.path.isfile(csv_path):
                    m = _pd.read_csv(csv_path, usecols=["rir_id","filename"])
                    row = m.loc[m["rir_id"]==rid]
                    if len(row):
                        member2 = str(row.iloc[0]["filename"])
                        if member2 in f:
                            obj = f[member2]
                            if isinstance(obj, h5py.Dataset):
                                return float(sr), _ds_to_ir(obj)
                            if isinstance(obj, h5py.Group):
                                for k in ("rir","RIR","ir","IR"):
                                    if k in obj and isinstance(obj[k], h5py.Dataset):
                                        return float(sr), _ds_to_ir(obj[k])
                                for k in obj.keys():
                                    if isinstance(obj[k], h5py.Dataset) and obj[k].dtype.kind in "fF":
                                        return float(sr), _ds_to_ir(obj[k])
            except Exception:
                pass
            # packed 2D/3D dataset fallback (rows = RIRs)
            cands = []
            for k in f.keys():
                obj = f[k]
                if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fF" and obj.ndim >= 2:
                    cands.append(obj)
            for gname in ("rirs","RIRs","rir","RIR"):
                g = f.get(gname)
                if g is not None:
                    for k in g.keys():
                        obj = g[k]
                        if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fF" and obj.ndim >= 2:
                            cands.append(obj)
            for ds in cands:
                if ds.shape[0] >= (rid or 0) and (rid or 0) > 0:
                    return float(sr), _to_1d(ds[rid-1])
            raise KeyError(f"dEchorate: RIR id {rid} not found")

        # 3) filename-like member: exact/fuzzy
        if member:
            token = member.split("/")[-1]
            if token in f:
                obj = f[token]
                if isinstance(obj, h5py.Dataset): return float(sr), _ds_to_ir(obj)
                if isinstance(obj, h5py.Group):
                    for k in ("rir","RIR","ir","IR"):
                        if k in obj and isinstance(obj[k], h5py.Dataset):
                            return float(sr), _ds_to_ir(obj[k])
                    for sub in obj.keys():
                        if isinstance(obj[sub], h5py.Dataset) and obj[sub].dtype.kind in "fF":
                            return float(sr), _ds_to_ir(obj[sub])
            for k in f.keys():
                if token in k:
                    obj = f[k]
                    if isinstance(obj, h5py.Dataset): return float(sr), _ds_to_ir(obj)
                    if isinstance(obj, h5py.Group):
                        for cand in ("rir","RIR","ir","IR"):
                            if cand in obj and isinstance(obj[cand], h5py.Dataset):
                                return float(sr), _ds_to_ir(obj[cand])
                        for sub in obj.keys():
                            if isinstance(obj[sub], h5py.Dataset) and obj[sub].dtype.kind in "fF":
                                return float(sr), _ds_to_ir(obj[sub])
            raise KeyError(f"HDF5 member not found: {member}")

        # 4) no member: return first float dataset we see
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fF":
                return float(sr), _ds_to_ir(obj)
        raise KeyError("No RIR dataset found in HDF5")
# ---------------- T60 estimator ----------------
def estimate_t60(sr, x):
    x = x.astype(np.float32)
    if len(x) < max(1, int(sr//10)):
        return None, "too_short"
    e = x**2
    edc = np.flip(np.cumsum(np.flip(e)))
    if edc[0] == 0:
        return None, "zero_energy"
    edc /= edc[0]
    edc_db = 10*np.log10(np.maximum(edc, 1e-12))

    def fit(lo, hi):
        idx = np.where((edc_db >= hi) & (edc_db <= lo))[0]
        if idx.size < 50:
            return None
        t = idx/float(sr); y = edc_db[idx]
        A = np.vstack([t, np.ones_like(t)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        if m >= -1e-6:
            return None
        return -60.0/m

    T30 = fit(-5, -35)
    if T30 and 0.05 <= T30 <= 20:
        return float(T30), "T30"
    T20 = fit(-5, -25)
    if T20 and 0.05 <= T20 <= 20:
        return float(1.5*T20), "T20->T60"
    return None, "no_fit"

# ---------------- clarity metrics ----------------
def _clarity(sr, x, ms):
    import numpy as _np
    n = int(round(ms/1000.0*sr))
    if n <= 0 or len(x) <= n:
        return None
    e = _np.square(x.astype(_np.float32))
    early = float(e[:n].sum())
    late  = float(e[n:].sum())
    if late <= 0:
        return None
    return 10.0 * _np.log10((early + 1e-12)/(late + 1e-12))

def clarity_db(sr, x):
    c50 = _clarity(sr, x, 50)
    c80 = _clarity(sr, x, 80)
    return c50, c80

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Backfill T60 for WAV/SOFA/MAT entries in rirs.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--zip-roots", default="")
    ap.add_argument("--only-data-kind", default=None,
                   help="comma-separated list of data_kind values to include (e.g. rir,brir)")
    args = ap.parse_args()

    roots = [p for p in args.zip_roots.split(";") if p and os.path.isdir(p)]
    for d in (os.path.expanduser("~/Database/data"), os.path.dirname(args.db)):
        if d and os.path.isdir(d) and d not in roots:
            roots.append(d)

    con = sqlite3.connect(args.db); migrate(con); cur = con.cursor()
    where = ["t60_s IS NULL"]
    params = []
    if args.dataset:
        where.append("dataset=?"); params.append(args.dataset)
    if args.only_data_kind:
        kinds = [k.strip() for k in args.only_data_kind.split(",") if k.strip()]
        if kinds:
            where.append("COALESCE(data_kind, '') IN ({})".format(",".join(["?"]*len(kinds))))
            params.extend(kinds)
    if args.dataset:
        where.append("dataset=?"); params.append(args.dataset)
    rows = cur.execute(
        "SELECT id,file_path,file_format,file_name FROM rirs WHERE {} LIMIT ?"
        .format(" AND ".join(where)), (*params, args.limit)
    ).fetchall()

    ok = fail = 0
    for rid, fpath, ffmt, fname in rows:
        try:
            if not fpath:
                raise FileNotFoundError("empty file_path")
            fmt = (ffmt or "").upper()
            low = fpath.lower()

            # Decide loader
            if "::" in fpath:
                base, member = fpath.split("::", 1)
                # HDF5 container?
                if base.lower().endswith((".h5",".hdf5")) or (ffmt or "").upper()=="HDF5":
                    sr, x = read_dechorate_from_hdf5(base, member)
                else:
                    zip_name = base
                    zp = None
                for r in roots:
                    p = Path(r).joinpath(zip_name)
                    if p.exists():
                        zp = str(p); break
                if zp is None:
                    # allow absolute path
                    from pathlib import Path as _P
                    if _P(zip_name).exists():
                        zp = zip_name
                    else:
                        raise FileNotFoundError("ZIP not found: {}".format(zip_name))
                if member.lower().endswith(".sofa") or fmt in ("ZIP-SOFA","SOFA"):
                    sr, x = read_sofa_from_zip(zp, member)
                elif member.lower().endswith(".mat") or fmt in ("ZIP-MAT","MAT"):
                    sr, x = read_mat_from_zip(zp, member)
                else:
                    sr, x = read_wav_from_zip(zp, member)
            else:
                if low.endswith((".h5",".hdf5")) or fmt == "HDF5":
                    sr, x = read_dechorate_from_hdf5(fpath, None)
                elif low.endswith(".sofa") or fmt == "SOFA":
                    sr, x = read_sofa_from_fs(fpath)
                elif low.endswith(".mat") or fmt == "MAT":
                    sr, x = read_mat_from_fs(fpath)
                else:
                    sr, x = read_wav_from_fs(fpath)

            t60, method = estimate_t60(sr, x)
            c50, c80 = clarity_db(sr, x)
            # annotate if fs was guessed from name
            if method and "->" not in method and "T30" in method:
                pass  # keep simple
            # If fs came from filename guess, record it
            if "binaural" in fname.lower() or "phone" in fname.lower():
                # Check: did we have to guess fs? (no direct way here, but ok to annotate)
                pass

            if t60 is not None:
                cur.execute("UPDATE rirs SET t60_s=?, t60_method=? WHERE id=?", (float(t60), method, rid))
                ok += 1
                print("[OK] {} -> {:.3f}s ({})".format(fname, t60, method))
            else:
                cur.execute("UPDATE rirs SET t60_method=? WHERE id=?", (method, rid))
                print("[..] {} -> {}".format(fname, method))
        except Exception as e:
            fail += 1
            cur.execute("UPDATE rirs SET t60_method=? WHERE id=?", ("err:{}:{}".format(type(e).__name__, e), rid))
            print("[ERR] {} -> {}".format(fname, e), file=sys.stderr)

    con.commit(); con.close()
    print("Backfilled: OK={}, FAIL={}".format(ok, fail))

if __name__ == "__main__":
    main()
