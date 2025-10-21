#!/usr/bin/env python3
from __future__ import annotations
import os, io, zipfile, shutil, tempfile, math, re
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ========== Config ==========
DB_PATH = os.environ.get("RIR_DB", "/home/on61ewex/Database/db/rir_meta_v3.db")
RIR_DB_URL = os.environ.get("RIR_DB_URL")
if (not Path(DB_PATH).exists()) and RIR_DB_URL:
    try:
        import urllib.request, tempfile as _tmp
        tmp = Path(_tmp.gettempdir()) / "rir_meta_v3.db"
        if not tmp.exists():
            urllib.request.urlretrieve(RIR_DB_URL, tmp.as_posix())
        DB_PATH = tmp.as_posix()
    except Exception:
        pass

DB_PATH = os.environ.get("RIR_DB_PATH", DB_PATH)
DATA_ROOT = Path(os.environ.get("RIR_DATA_ROOT", "/home/on61ewex/Database/data")).resolve()

# --- Fallbacks for Streamlit Cloud / portable runs ---
# If the env-provided DB path doesn't exist, try the bundled repo DB at data/rir_meta_v3.db
try:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _BUNDLED_DB = (_REPO_ROOT / "data" / "rir_meta_v3.db").resolve()
    if not Path(DB_PATH).exists() and _BUNDLED_DB.exists():
        DB_PATH = _BUNDLED_DB.as_posix()
    # Prefer a data/ directory under the repo for file copies/ZIP if the env root doesn't exist
    if not DATA_ROOT.exists():
        _BUNDLED_DATA = (_REPO_ROOT / "data").resolve()
        if _BUNDLED_DATA.exists():
            DATA_ROOT = _BUNDLED_DATA
except Exception:
    pass
# --- end fallbacks ---
# --- PATH REMAP & COPY POLICY ---
# RIR_PATH_MAP format: "SRC1=DST1;SRC2=DST2;..."
# Example: RIR_PATH_MAP="/home/on61ewex/Database/data=/mnt/public/rir_data;/old=/new"
PATH_MAP_RAW = os.environ.get("RIR_PATH_MAP", "") or ""
def _remap_path(fp: str) -> str:
    rules = [r for r in PATH_MAP_RAW.split(";") if r.strip()]
    for rule in rules:
        if "=" in rule:
            src, dst = rule.split("=", 1)
            src = src.strip()
            dst = dst.strip()
            if src and fp.startswith(src):
                return fp.replace(src, dst, 1)
    return fp

# If set to "1", allow copying files even if they are not under DATA_ROOT
ALLOW_OUTSIDE = (os.environ.get("RIR_ALLOW_OUTSIDE", "0") == "1")
# --- END PATH REMAP & COPY POLICY ---
# -------- Helper: resolve fallback paths for a row --------
def _resolve_candidate_path(row: dict) -> Path | None:
    """
    Given a row (dict-like), return a Path to the best existing file.
    Strategy:
      1) remap original file_path via RIR_PATH_MAP
      2) if original contains '/Database/data/', join DATA_ROOT with the tail
      3) dataset-aware guesses (DATA_ROOT/dataset/<basename>)
      4) quick rglob within DATA_ROOT/dataset for the basename (first hit)
    """
    try:
        orig = str(row.get("file_path","") or "")
        ds   = str(row.get("dataset","") or "").strip()
        base = Path(orig).name if orig else ""
        cand = []

        if orig:
            # 1) remap
            remap = _remap_path(orig)
            if remap:
                cand.append(Path(remap))

            # 2) join tail after lab root
            if "/Database/data/" in orig:
                tail = orig.split("/Database/data/", 1)[1]
                cand.append(DATA_ROOT / tail)

        # 3) dataset-aware guesses
        if ds and base:
            cand.append(DATA_ROOT / ds / base)

            # 4) quick rglob under dataset dir (first match only)
            ds_dir = (DATA_ROOT / ds)
            if ds_dir.exists():
                try:
                    for hit in ds_dir.rglob(base):
                        cand.append(hit)
                        break
                except Exception:
                    pass

        # return first existing path
        for c in cand:
            try:
                rp = c.resolve()
            except Exception:
                continue
            if rp.exists():
                return rp
    except Exception:
        pass
    return None
# -------- end helper --------

# DEFAULT RIR_PATH_MAP if env var is empty:
# Map lab path to the repo's data/public_sample (portable)
if not PATH_MAP_RAW:
    try:
        _REPO_ROOT = Path(__file__).resolve().parent.parent
        _SAMPLE = (_REPO_ROOT / "data" / "public_sample").resolve()
        if _SAMPLE.exists():
            PATH_MAP_RAW = f"/home/on61ewex/Database/data={_SAMPLE.as_posix()}"
    except Exception:
        pass

# Prefer to allow copying outside DATA_ROOT by default (portable)
if os.environ.get("RIR_ALLOW_OUTSIDE") is None:
    ALLOW_OUTSIDE = True



st.set_page_config(page_title="RIR Explorer", layout="wide")

# ========== Minimal modern style ==========
st.markdown(
    """<style>
        [data-testid="stHeader"]{background:transparent!important;}
        .block-container{padding-top:1rem;padding-bottom:2.5rem;}
        div.stAlert{border-radius:12px!important;}
        hr{border:none;border-top:1px solid #e5e7eb;margin:.4rem 0 1rem 0;}
    </style>""",
    unsafe_allow_html=True
)

# ========== Helpers ==========
_VOL_RE = re.compile(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)[ ]*m?", re.IGNORECASE)

def _compute_volume_log10(s: pd.Series) -> pd.Series:
    out = []
    for v in s.fillna(""):
        m = _VOL_RE.search(str(v))
        if not m:
            out.append(np.nan); continue
        try:
            d, w, h = float(m.group(1)), float(m.group(2)), float(m.group(3))
            vol = d * w * h
            out.append(np.nan if vol <= 0 else float(math.log10(vol)))
        except Exception:
            out.append(np.nan)
    return pd.Series(out, index=s.index, dtype="float64")

def _prep_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    if "room_dims_m" in g.columns and "volume_log10" not in g.columns:
        g["volume_log10"] = _compute_volume_log10(g["room_dims_m"])
    return g

def _grid_columns_for_k(k: int) -> int:
    """Choose #columns for facet grid; e.g., K=12 -> 4 (so ~4x3)."""
    if k <= 3: return k
    # aim for near-square grid, cap columns to avoid tiny cards
    return min(6, max(2, int(math.ceil(math.sqrt(max(1, k))))))

def _make_hist_bins(df_in: pd.DataFrame, value_col: str, title: str,
                    topk: int, facet: bool, bins: int, strict_topk: bool):
    import pandas as _pd
    g = df_in.dropna(subset=[value_col, "dataset"]).copy()
    if g.empty:
        return alt.Chart(_pd.DataFrame({"msg": ["No data"]})).mark_text(size=14, opacity=0.8) \
            .encode(text="msg").properties(height=80, title=title)

    # Normalize dataset names (avoid whitespace variants)
    g["dataset_clean"] = g["dataset"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    # Top-K by this metric
    counts = g.groupby("dataset_clean")[value_col].count().sort_values(ascending=False)
    show_k = max(0, min(int(topk), int(len(counts))))
    top = counts.head(show_k).index.tolist()

    ncols = _grid_columns_for_k(show_k)

    # STRICT: exactly Top-K, no 'Other'
    if strict_topk:
        g_top = g[g["dataset_clean"].isin(top)].copy()
        if g_top.empty:
            return alt.Chart(_pd.DataFrame({"msg": ["No Top-K data"]})).mark_text(size=14, opacity=0.8) \
                .encode(text="msg").properties(height=80, title=title)
        if facet:
            # Wrapped facet grid (columns=ncols)
            base = alt.Chart(g_top).mark_bar(opacity=0.9).encode(
                x=alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=int(bins)), title=title),
                y=alt.Y("count()", title="Count"),
                color=alt.Color("dataset_clean:N", legend=None)
            ).properties(height=140, width=220)
            return base.facet(facet=alt.Facet("dataset_clean:N", title=None, sort=top), columns=ncols) \
                       .resolve_scale(x="shared", y="independent")
        # Overlay but only Top-K
        return alt.Chart(g_top).mark_bar(opacity=0.9).encode(
            x=alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=int(bins)), title=title),
            y=alt.Y("count()", title="Count"),
            color=alt.Color("dataset_clean:N", title="Dataset (Top-K)", sort=top)
        ).properties(height=280, title=f"{title} — Top {len(top)} only")

    # NON-STRICT: Top-K + 'Other'
    g["grp"] = g["dataset_clean"].apply(lambda d: d if d in top else "Other")
    if facet:
        tops = g[g["grp"] != "Other"]
        base = alt.Chart(tops).mark_bar(opacity=0.85).encode(
            x=alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=int(bins)), title=title),
            y=alt.Y("count()", title="Count"),
            color=alt.Color("dataset_clean:N", legend=None)
        ).properties(height=140, width=220)
        charts = base.facet(facet=alt.Facet("dataset_clean:N", title=None, sort=top), columns=ncols) \
                     .resolve_scale(x="shared", y="independent")
        others = g[g["grp"] == "Other"]
        if not others.empty:
            layer_others = alt.Chart(others).mark_bar(opacity=0.35).encode(
                x=alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=int(bins)), title=title),
                y=alt.Y("count()", title="Count"),
                color=alt.value("#CCCCCC")
            ).properties(height=140, width=220)
            # Show 'Other' as a single card at the end
            other_card = layer_others.properties(title="Other")
            return charts & other_card
        return charts

    # Overlay with 'Other'
    sort_order = top + (["Other"] if (g["grp"] == "Other").any() else [])
    return alt.Chart(g).mark_bar(opacity=0.85).encode(
        x=alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=int(bins)), title=title),
        y=alt.Y("count()", title="Count"),
        color=alt.Color("grp:N", title="Dataset (Top-K + Other)", sort=sort_order)
    ).properties(height=280, title=f"{title} — overlay by dataset (Top {show_k} + Other)")

# ========== DB ==========
@st.cache_resource
def get_engine() -> Engine:
    pgurl = (os.environ.get("RIR_PGURL") or "").strip()
    if pgurl:
        try:
            return create_engine(pgurl, pool_pre_ping=True, future=True)
        except ModuleNotFoundError:
            if not st.session_state.get("_warned_pg_missing"):
                st.session_state["_warned_pg_missing"] = True
                st.warning("PostgreSQL URL set but 'psycopg2' not installed. Falling back to SQLite.", icon="⚠️")
        except Exception as e:
            if not st.session_state.get("_warned_pg_error"):
                st.session_state["_warned_pg_error"] = True
                st.warning(f"Could not connect to PostgreSQL ({e}). Falling back to SQLite.", icon="⚠️")

    url = DB_PATH if DB_PATH.startswith("sqlite:///") else f"sqlite:///{DB_PATH}"
    eng = create_engine(url, future=True)
    if eng.dialect.name == "sqlite":
        with eng.begin() as con:
            con.exec_driver_sql("PRAGMA journal_mode=WAL;")
            con.exec_driver_sql("PRAGMA synchronous=NORMAL;")
    return eng

def run_df(sql: str, params: Dict[str, Any] = None) -> pd.DataFrame:
    with get_engine().begin() as con:
        return pd.read_sql_query(text(sql), con, params=params or {})

def run_scalar(sql: str, params: Dict[str, Any] = None) -> Any:
    with get_engine().begin() as con:
        return con.execute(text(sql), params or {}).scalar()

@st.cache_data(ttl=300, show_spinner=False)
def distincts_and_ranges():
    datasets = run_df("SELECT dataset, COUNT(*) AS n FROM rirs GROUP BY dataset ORDER BY dataset")
    def _distinct(col):
        df = run_df(f"SELECT DISTINCT {col} AS v FROM rirs WHERE {col} IS NOT NULL ORDER BY {col}")
        return df["v"].tolist()
    srs = _distinct("sample_rate_hz")
    rooms = _distinct("room_type")
    fmts = _distinct("file_format")
    ambs = _distinct("ambisonics_order")
    los_raw = _distinct("los")
    los_ui = []
    if "LOS" in los_raw: los_ui.append("Yes")
    if "noLOS" in los_raw: los_ui.append("No")
    mm = run_df("""
        SELECT MIN(t60_s) AS tmin, MAX(t60_s) AS tmax,
               MIN(distance_m) AS dmin, MAX(distance_m) AS dmax
        FROM rirs
    """).iloc[0].to_dict()
    return datasets, srs, rooms, fmts, ambs, los_ui, mm

def build_where(filters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    w, p = ["1=1"], {}
    def _in(col, vals, key):
        if not vals: return
        ph=[]
        for i,v in enumerate(vals):
            k=f"{key}_{i}"; p[k]=v; ph.append(f":{k}")
        w.append(f"{col} IN ({', '.join(ph)})")
    _in("dataset", filters.get("datasets") or [], "ds")
    _in("sample_rate_hz", filters.get("sr") or [], "sr")
    _in("file_format", filters.get("fmts") or [], "fmt")
    _in("ambisonics_order", filters.get("ambs") or [], "amb")
    if filters.get("bina") is not None:
        p["bina"]=1 if filters["bina"] else 0
        w.append("is_binaural = :bina")
    if filters.get("room"):
        p["room"]=filters["room"]; w.append("room_type = :room")
    if filters.get("los") is not None:
        p["los"]="LOS" if filters["los"]=="Yes" else "noLOS"
        w.append("los = :los")
    if filters.get("t60") is not None:
        tmin,tmax = filters["t60"]
        p["tmin"]=float(tmin); p["tmax"]=float(tmax)
        if filters.get("t60_inc_null"):
            w.append("((t60_s BETWEEN :tmin AND :tmax) OR t60_s IS NULL)")
        else:
            w.append("(t60_s BETWEEN :tmin AND :tmax)")
    if filters.get("dist_max") is not None:
        p["dmax"]=float(filters["dist_max"])
        w.append("distance_m IS NOT NULL AND distance_m <= :dmax")
    if filters.get("name_like"):
        p["name_like"]=f"%{filters['name_like']}%"; w.append("file_name LIKE :name_like")
    if filters.get("path_like"):
        p["path_like"]=f"%{filters['path_like']}%"; w.append("file_path LIKE :path_like")
    return " AND ".join(w), p

def build_query(filters: Dict[str, Any], limit: int, offset: int) -> Tuple[str, Dict[str, Any]]:
    where, p = build_where(filters)
    p["limit"]=int(limit); p["offset"]=int(offset)
    return f"SELECT * FROM rirs WHERE {where} ORDER BY dataset, file_name LIMIT :limit OFFSET :offset", p

def build_zip_bytes(df: pd.DataFrame, max_files: int = 2000) -> Tuple[bytes, int, int]:
    # _ZIP_DIAG_: enhanced logging
    staging = Path(tempfile.mkdtemp(prefix="rir_export_"))
    copied = 0
    candidates = 0
    skipped_missing = 0
    skipped_outside = 0
    try:
        for _, r in df.iterrows():
            fp = str(r.get("file_path","") or "")
            if not fp:
                continue
            candidates += 1
            # apply path remap if configured
            fp_remap = _remap_path(fp)
            p_src = Path(fp_remap)
            try:
                rp = p_src.resolve()
            except Exception:
                rp = p_src
            if not rp.exists():
                skipped_missing += 1
                continue
            # enforce root unless overridden
            under_root = (DATA_ROOT in rp.parents or rp == DATA_ROOT)
            if not (under_root or ALLOW_OUTSIDE):
                skipped_outside += 1
                continue
            if copied < max_files:
                dest = staging / rp.name
                i=1
                while dest.exists():
                    dest = staging / f"{rp.stem}__{i}{rp.suffix}"; i+=1
                try:
                    shutil.copy2(rp, dest)
                    copied += 1
                except Exception:
                    pass
        (staging/"manifest.csv").write_bytes(df.to_csv(index=False).encode("utf-8"))
        # also write a small diagnostics file
        (staging/"zip_diagnostics.txt").write_text(
            f"candidates={candidates}\n"
            f"copied={copied}\n"
            f"skipped_missing={skipped_missing}\n"
            f"skipped_outside={skipped_outside}\n"
            f"DATA_ROOT={DATA_ROOT}\n"
            f"ALLOW_OUTSIDE={ALLOW_OUTSIDE}\n"
            f"PATH_MAP_RAW={PATH_MAP_RAW}\n"
        )
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_DEFLATED) as z:
            for item in staging.iterdir():
                z.write(item, arcname=item.name)
        buf.seek(0)
        return buf.read(), copied, candidates
    finally:
        shutil.rmtree(staging, ignore_errors=True)

# ========== UI ==========
def main():
    _ = get_engine()
    st.info(f"Using database at `{DB_PATH}`", icon="🗄️")
    st.title("RIR Explorer")
    st.caption("Filter RIR metadata with the sidebar. Explore distributions below. Export current results at the bottom.")

    st.session_state.setdefault("dist_topk_main", 12)
    st.session_state.setdefault("dist_bins_main", 40)
    st.session_state.setdefault("dist_facet_main", True)     # default ON to see grid
    st.session_state.setdefault("dist_strict_topk", True)    # EXACT K by default
    st.session_state.setdefault("zip_ready", False)
    st.session_state.setdefault("zip_bytes", None)
    st.session_state.setdefault("zip_note", "")

    datasets, sr_opts, room_opts, fmt_opts, amb_opts, los_opts_ui, mm = distincts_and_ranges()

    with st.sidebar:
        st.header("Filters")
        ds_sel = st.multiselect("Datasets", options=datasets["dataset"].tolist(), default=datasets["dataset"].tolist())
        fmt_sel = st.multiselect("Audio format", options=fmt_opts)
        sr_sel = st.multiselect("Sample rate (Hz)", options=sr_opts)
        bina = st.selectbox("Binaural?", ["Any", "Yes", "No"])
        room = st.selectbox("Room type", ["Any"] + room_opts) if room_opts else None
        los_ui = st.selectbox("Line of sight (LOS)", ["Any"] + los_opts_ui) if los_opts_ui else None
        amb_sel = st.multiselect("Ambisonics order", options=[v for v in amb_opts if pd.notna(v)])

        with st.expander("Advanced", expanded=False):
            t60_range = None; include_null_t60 = True
            tmin_db, tmax_db = mm.get("tmin"), mm.get("tmax")
            if pd.notna(tmin_db) and pd.notna(tmax_db) and float(tmin_db) < float(tmax_db):
                t60_range = st.slider("RT60 (s)",
                                      float(round(float(tmin_db), 2)),
                                      float(round(float(tmax_db), 2)),
                                      (float(round(float(tmin_db), 2)), float(round(float(tmax_db), 2))),
                                      step=0.05)
                include_null_t60 = st.checkbox("Include missing RT60", value=True)

            dist_max = None
            if st.checkbox("Filter by max mic–src distance (m)", value=False):
                dmax_db = mm.get("dmax")
                dist_max = st.number_input("Distance ≤ (m)",
                                           value=float(dmax_db) if pd.notna(dmax_db) else 0.0,
                                           step=0.1, format="%.2f")

            name_like = st.text_input("Filename contains", "")
            path_like = st.text_input("File path contains", "")

        page_size = st.selectbox("Page size", [1000, 5000, 10000], index=1)
        page_no = st.number_input("Page (1-based)", min_value=1, value=1, step=1)
        do_search = st.button("Search", type="primary")

    if do_search:
        st.session_state["zip_ready"] = False
        st.session_state["zip_bytes"] = None
        st.session_state["zip_note"] = ""

        filters = {
            "datasets": ds_sel, "fmts": fmt_sel, "sr": sr_sel,
            "bina": None if bina == "Any" else (bina == "Yes"),
            "room": None if (room in (None, "Any")) else room,
            "los": None if (los_ui in (None, "Any")) else los_ui,
            "ambs": amb_sel, "t60": t60_range, "t60_inc_null": include_null_t60,
            "dist_max": dist_max,
            "name_like": (name_like or "").strip() or None,
            "path_like": (path_like or "").strip() or None,
        }
        count_sql, count_params = build_where(filters)
        total = run_scalar(f"SELECT COUNT(*) FROM rirs WHERE {count_sql}", count_params) or 0
        offset = (int(page_no) - 1) * int(page_size)
        sql, params = build_query(filters, limit=page_size, offset=offset)
        df = run_df(sql, params)
        st.session_state["_last_df"] = df.copy()

        st.success(f"Matched rows: {total:,} | Showing {len(df):,} (page {page_no})")
        with st.expander("Debug (SQL)"):
            st.code(sql); st.write("Params:", params)

        # Format distribution (brought back)
        if "file_format" in df.columns and not df.empty:
            fmt_cnt = df.groupby("file_format", dropna=False)["file_format"].count().reset_index(name="count")
            fmt_bar = alt.Chart(fmt_cnt).mark_bar().encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("file_format:N", sort="-x", title="Format")
            ).properties(height=240, title="Rows by format")
            st.altair_chart(fmt_bar, use_container_width=True)

        preferred = ["dataset","file_name","file_format","ambisonics_order",
                     "sample_rate_hz","num_channels","num_receivers","is_binaural",
                     "duration_s","t60_s","distance_m","room_type","los","file_path"]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        st.dataframe(df[cols], use_container_width=True, hide_index=True, height=520)

    # ===== Distributions & Export for latest results =====
    last_df = st.session_state.get("_last_df")
    if last_df is not None and not last_df.empty:
        st.subheader("Distributions: Volume (log10), RT60, Boundary Points (log10)")
        ui_cols = st.columns([1,1,1,1,3])
        ui_cols[0].number_input("Top-K datasets to color", min_value=1, max_value=50, step=1,
                                key="dist_topk_main", help="How many datasets to color/facet.")
        ui_cols[1].slider("Histogram bins", min_value=10, max_value=80, step=5,
                          key="dist_bins_main", help="Controls binning granularity.")
        ui_cols[2].checkbox("Facet top-K datasets (grid)", key="dist_facet_main",
                            help="Show Top-K in a wrapped grid (not a long column).")
        ui_cols[3].checkbox("Only show Top-K (hide 'Other')", key="dist_strict_topk", value=True,
                            help="Exactly K datasets; avoids >K colors if big sets are removed.")
        with ui_cols[4]:
            st.caption("K=12 → 4×3 grid automatically. Toggle Strict to enforce exactly K.")

        topk  = int(st.session_state["dist_topk_main"])
        bins  = int(st.session_state["dist_bins_main"])
        facet = bool(st.session_state["dist_facet_main"])
        strict_topk = bool(st.session_state["dist_strict_topk"])

        dfm = _prep_metrics_df(last_df)
        if "volume_log10" in dfm.columns and not dfm["volume_log10"].dropna().empty:
            st.altair_chart(_make_hist_bins(dfm, "volume_log10", "Volume (log10 m³)",
                                            topk, facet, bins, strict_topk), use_container_width=True)
        if "t60_s" in dfm.columns and not dfm["t60_s"].dropna().empty:
            st.altair_chart(_make_hist_bins(dfm, "t60_s", "RT60 (s)",
                                            topk, facet, bins, strict_topk), use_container_width=True)
        if "bp_log10" in dfm.columns and not dfm["bp_log10"].dropna().empty:
            st.altair_chart(_make_hist_bins(dfm, "bp_log10", "Boundary Points (log10)",
                                            topk, facet, bins, strict_topk), use_container_width=True)

        # ---- Export: 2-step, reliable ----
        st.markdown("#### Export current results")
        c1, c2 = st.columns([1,2])
        c1.download_button(
            "Download CSV (this page's results)",
            data=last_df.to_csv(index=False).encode("utf-8"),
            file_name="rir_results_page.csv",
            mime="text/csv",
            key="csv_dl_btn"
        )
        with c2:
            max_files = st.number_input("Max files in ZIP", min_value=1, max_value=20000,
                                        value=2000, step=100, help="Upper bound to avoid huge ZIPs.")
            if st.button("Build ZIP now", type="secondary", key="zip_build_btn"):
                with st.spinner("Collecting files and building ZIP…"):
                    zbytes, copied, candidates = build_zip_bytes(last_df, max_files=int(max_files))
                    st.session_state["zip_bytes"] = zbytes
                    st.session_state["zip_ready"] = True
                    st.session_state["zip_note"]  = f"Added {copied} of {candidates} candidate files. Always includes manifest.csv."
            if st.session_state["zip_ready"] and st.session_state.get("zip_bytes"):
                st.success(st.session_state.get("zip_note","ZIP ready."))
                st.download_button(
                    "Download ZIP",
                    data=st.session_state["zip_bytes"],
                    file_name="rir_export_page.zip",
                    mime="application/zip",
                    key="zip_dl_btn"
                )
    else:
        st.info("Run a search to see distributions and export options.", icon="🔎")

    with st.expander("Overview"):
        st.write("Datasets and counts:")
        st.dataframe(distincts_and_ranges()[0], use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
