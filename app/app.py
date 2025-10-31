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

# ---- PNG/SVG/PDF export via vl-convert (no browser, no chrome) ----
try:
    from vl_convert import vegalite_to_png, vegalite_to_svg, vegalite_to_pdf
except Exception:
    vegalite_to_png = vegalite_to_svg = vegalite_to_pdf = None

# Ensure fallback cache dir exists if used
try:
    Path(os.environ.get("TMPDIR", (Path.home()/".cache/rir_tmp").as_posix())).mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# ========== Config ==========
DB_PATH = os.environ.get("RIR_DB", "/home/on61ewex/Database/db/rir_meta_v3.db")
DB_PATH = os.environ.get("RIR_DB_PATH", DB_PATH)
DATA_ROOT = Path(os.environ.get("RIR_DATA_ROOT", "/home/on61ewex/Database/data")).resolve()

# --- PATH REMAP & COPY POLICY ---
PATH_MAP_RAW = os.environ.get("RIR_PATH_MAP", "") or ""
def _remap_path(fp: str) -> str:
    rules = [r for r in PATH_MAP_RAW.split(";") if r.strip()]
    for rule in rules:
        if "=" in rule:
            src, dst = rule.split("=", 1)
            src = src.strip(); dst = dst.strip()
            if src and fp.startswith(src):
                return fp.replace(src, dst, 1)
    return fp

ALLOW_OUTSIDE = (os.environ.get("RIR_ALLOW_OUTSIDE", "0") == "1")
if os.environ.get("RIR_ALLOW_OUTSIDE") is None:
    ALLOW_OUTSIDE = True

PUBLIC_DATA_BASE_URL = (os.environ.get("PUBLIC_DATA_BASE_URL","") or "").rstrip("/")

def path_to_public_url(fp: str) -> str | None:
    if not PUBLIC_DATA_BASE_URL:
        return None
    try:
        fp2 = _remap_path(str(fp or ""))
        p = Path(fp2)
        rp = p.resolve() if p.exists() else p
        if DATA_ROOT in rp.parents or rp == DATA_ROOT:
            rel = rp.relative_to(DATA_ROOT)
            return f"{PUBLIC_DATA_BASE_URL}/{rel.as_posix()}"
        lab_root = Path("/home/on61ewex/Database/data")
        if lab_root in rp.parents or rp == lab_root:
            rel = rp.relative_to(lab_root)
            return f"{PUBLIC_DATA_BASE_URL}/{rel.as_posix()}"
    except Exception:
        return None
    return None

# -------- Helper: resolve fallback paths for a row --------
def _resolve_candidate_path(row: dict) -> Path | None:
    try:
        orig = str(row.get("file_path","") or "")
        ds   = str(row.get("dataset","") or "").strip()
        base = Path(orig).name if orig else ""
        cand = []
        if orig:
            remap = _remap_path(orig)
            if remap: cand.append(Path(remap))
            if "/Database/data/" in orig:
                tail = orig.split("/Database/data/", 1)[1]
                cand.append(DATA_ROOT / tail)
        if ds and base:
            cand.append(DATA_ROOT / ds / base)
            ds_dir = (DATA_ROOT / ds)
            if ds_dir.exists():
                for hit in ds_dir.rglob(base):
                    cand.append(hit); break
        for c in cand:
            try: rp = c.resolve()
            except Exception: continue
            if rp.exists(): return rp
    except Exception:
        pass
    return None
# -------- end helper --------

st.set_page_config(page_title="RIR Explorer", layout="wide")
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
import json
_VOL_RE = re.compile(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)[ ]*m?", re.IGNORECASE)

def _compute_volume_log10(s: pd.Series) -> pd.Series:
    out = []
    for v in s.fillna(""):
        m = _VOL_RE.search(str(v))
        if not m: out.append(np.nan); continue
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
    if k <= 3: return k
    return min(6, max(2, int(math.ceil(math.sqrt(max(1, k))))))

def _make_hist_bins_v2(df_in: pd.DataFrame, value_col: str, title: str,
                    topk: int, facet: bool, bins: int, strict_topk: bool):
    import pandas as _pd
    g = df_in.dropna(subset=[value_col, "dataset"]).copy()
    if g.empty:
        return alt.Chart(_pd.DataFrame({"msg": ["No data"]})).mark_text(size=14, opacity=0.8) \
            .encode(text="msg").properties(height=80, title=title)

    g["dataset_clean"] = g["dataset"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    counts = g.groupby("dataset_clean")[value_col].count().sort_values(ascending=False)
    show_k = max(0, min(int(topk), int(len(counts))))
    top = counts.head(show_k).index.tolist()
    ncols = _grid_columns_for_k(show_k)

    if strict_topk:
        g_top = g[g["dataset_clean"].isin(top)].copy()
        if g_top.empty:
            return alt.Chart(_pd.DataFrame({"msg": ["No Top-K data"]})).mark_text(size=14, opacity=0.8) \
                .encode(text="msg").properties(height=80, title=title)
        if facet:
            base = alt.Chart(g_top).mark_bar(opacity=0.9).encode(
                x=alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=int(bins)), title=title),
                y=alt.Y("count()", title="Count"),
                color=alt.Color("dataset_clean:N", legend=None)
            ).properties(height=140, width=220)
            return base.facet(facet=alt.Facet("dataset_clean:N", title=None, sort=top), columns=ncols) \
                       .resolve_scale(x="shared", y="independent")
        return alt.Chart(g_top).mark_bar(opacity=0.9).encode(
            x=alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=int(bins)), title=title),
            y=alt.Y("count()", title="Count"),
            color=alt.Color("dataset_clean:N", title="Dataset (Top-K)", sort=top)
        ).properties(height=280, title=f"{title} — Top {len(top)} only")

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
            other_card = layer_others.properties(title="Other")
            return charts & other_card
        return charts

    sort_order = top + (["Other"] if (g["grp"] == "Other").any() else [])
    return alt.Chart(g).mark_bar(opacity=0.85).encode(
        x=alt.X(f"{value_col}:Q", bin=alt.Bin(maxbins=int(bins)), title=title),
        y=alt.Y("count()", title="Count"),
        color=alt.Color("grp:N", title="Dataset (Top-K + Other)", sort=sort_order)
    ).properties(height=280, title=f"{title} — overlay by dataset (Top {show_k} + Other)")

# ========== DB with cache key by file signature ==========
@st.cache_resource
def get_engine(db_url_key: str) -> Engine:
    eng = create_engine(db_url_key, future=True)
    if eng.dialect.name == "sqlite":
        with eng.begin() as con:
            con.exec_driver_sql("PRAGMA journal_mode=WAL;")
            con.exec_driver_sql("PRAGMA synchronous=NORMAL;")
    return eng

def _db_url() -> str:
    return DB_PATH if DB_PATH.startswith("sqlite:///") else f"sqlite:///{DB_PATH}"

def _db_sig() -> str:
    try:
        p = Path(DB_PATH); s = p.stat()
        return f"{p.as_posix()}::{int(s.st_mtime_ns)}::{s.st_size}"
    except Exception:
        return str(DB_PATH)

def run_df(sql: str, params: Dict[str, Any] = None) -> pd.DataFrame:
    with get_engine(_db_url()).begin() as con:
        return pd.read_sql_query(text(sql), con, params=params or {})

def run_scalar(sql: str, params: Dict[str, Any] = None) -> Any:
    with get_engine(_db_url()).begin() as con:
        return con.execute(text(sql), params or {}).scalar()

@st.cache_data(ttl=300, show_spinner=False)
def distincts_and_ranges(_sig: str):
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

# ---- ZIP builder (local) ----
def _best_local_path(row_like) -> Path | None:
    try:
        row = dict(row_like)
        fp = str(row.get("file_path","") or "")
        if fp:
            fp2 = _remap_path(fp)
            p = Path(fp2)
            try:
                rp = p.resolve()
            except Exception:
                rp = p
            if rp.exists():
                return rp
        rp2 = _resolve_candidate_path(row)
        if rp2 and rp2.exists():
            return rp2
    except Exception:
        pass
    return None

def build_zip_bytes(df: pd.DataFrame, max_files: int | None = None) -> Tuple[bytes, int, int]:
    staging = Path(tempfile.mkdtemp(prefix="rir_export_", dir=os.environ.get("TMPDIR", (Path.home()/".cache/rir_tmp").as_posix())))
    copied = candidates = skipped_missing = skipped_outside = 0
    tried_lines = []
    try:
        for _, r in df.iterrows():
            candidates += 1
            ds = str(r.get("dataset",""))
            raw_fp = str(r.get("file_path","") or "")
            rp = _best_local_path(r.to_dict())
            if not rp or not rp.exists():
                skipped_missing += 1
                tried_lines.append(f"MISSING | {ds} | {raw_fp}")
                continue
            under_root = (DATA_ROOT in rp.parents or rp == DATA_ROOT)
            if not (under_root or ALLOW_OUTSIDE):
                skipped_outside += 1
                tried_lines.append(f"OUTSIDE | {ds} | {rp}")
                continue
            tried_lines.append(f"COPY    | {ds} | {rp}")
            if (max_files is None) or (copied < max_files):
                dest = staging / rp.name
                i = 1
                while dest.exists():
                    dest = staging / f"{rp.stem}__{i}{rp.suffix}"
                    i += 1
                try:
                    shutil.copy2(rp, dest)
                    copied += 1
                except Exception as e:
                    tried_lines.append(f"ERROR   | {ds} | {rp} | {e}")

        # Always include manifest + diagnostics
        (staging / "manifest.csv").write_bytes(df.to_csv(index=False).encode("utf-8"))
        (staging / "zip_diagnostics.txt").write_text(
            "candidates={}\n"
            "copied={}\n"
            "skipped_missing={}\n"
            "skipped_outside={}\n"
            "DATA_ROOT={}\n"
            "ALLOW_OUTSIDE={}\n"
            "PATH_MAP_RAW={}\n"
            "{}\n".format(
                candidates, copied, skipped_missing, skipped_outside,
                DATA_ROOT, ALLOW_OUTSIDE, PATH_MAP_RAW, "\n".join(tried_lines)
            )
        )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for item in staging.iterdir():
                z.write(item, arcname=item.name)
        buf.seek(0)
        return buf.read(), copied, candidates
    finally:
        shutil.rmtree(staging, ignore_errors=True)
# ---- Chart export (robust) ----
def chart_bytes(chart: alt.Chart, fmt: str) -> Tuple[bytes, str]:
    spec = chart.to_dict()  # Vega-Lite spec
    if fmt == "json":
        return json.dumps(spec).encode("utf-8"), "application/json"
    if fmt == "html":
        return chart.to_html().encode("utf-8"), "text/html"
    # raster/vector via vl-convert if available
    if vegalite_to_png and fmt == "png":
        return vegalite_to_png(spec), "image/png"
    if vegalite_to_svg and fmt == "svg":
        return vegalite_to_svg(spec), "image/svg+xml"
    if vegalite_to_pdf and fmt == "pdf":
        return vegalite_to_pdf(spec), "application/pdf"
    raise RuntimeError("This format requires vl-convert-python. Try: pip install vl-convert-python")

# ========== UI ==========
def main():
    st.info(f"Using database at `{DB_PATH}`", icon="🗄️")
    st.title("RIR Explorer")
    st.caption("Filter RIR metadata with the sidebar. Explore distributions below. Export current results at the bottom.")

    st.session_state.setdefault("dist_topk_main", 12)
    st.session_state.setdefault("dist_bins_main", 40)
    st.session_state.setdefault("dist_facet_main", True)
    st.session_state.setdefault("dist_strict_topk", True)
    st.session_state.setdefault("zip_ready", False)
    st.session_state.setdefault("zip_bytes", None)
    st.session_state.setdefault("zip_note", "")

    datasets, sr_opts, room_opts, fmt_opts, amb_opts, los_opts_ui, mm = distincts_and_ranges(_db_sig())

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
                rng = (float(round(float(tmin_db), 2)), float(round(float(tmax_db), 2)))
                t60_range = st.slider("RT60 (s)", rng[0], rng[1], rng, step=0.05)
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
            "dist_max": dist_max, "name_like": (name_like or "").strip() or None,
            "path_like": (path_like or "").strip() or None,
        }
        # WHERE builder
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

        where = " AND ".join(w)
        total = run_scalar(f"SELECT COUNT(*) FROM rirs WHERE {where}", p) or 0
        offset = (int(page_no) - 1) * int(page_size)
        p["limit"]=int(page_size); p["offset"]=int(offset)
        sql = f"SELECT * FROM rirs WHERE {where} ORDER BY dataset, file_name LIMIT :limit OFFSET :offset"

        df = run_df(sql, p)
        if "file_path" in df.columns:
            try:
                df["public_url"] = df["file_path"].apply(path_to_public_url)
            except Exception:
                df["public_url"] = None

        st.session_state["_last_df"] = df.copy()

        st.success(f"Matched rows: {total:,} | Showing {len(df):,} (page {page_no})")
        with st.expander("Debug (SQL)"):
            st.code(sql); st.write("Params:", p)
            st.write("First 15 datasets:", run_df("SELECT dataset FROM rirs GROUP BY dataset ORDER BY dataset LIMIT 15")["dataset"].tolist())

        if "file_format" in df.columns and not df.empty:
            fmt_cnt = df.groupby("file_format", dropna=False)["file_format"].count().reset_index(name="count")
            fmt_bar = alt.Chart(fmt_cnt).mark_bar().encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("file_format:N", sort="-x", title="Format")
            ).properties(height=240, title="Rows by format")
            st.write("### Results")
            st.altair_chart(fmt_bar, use_container_width=True)

        preferred = ["dataset","file_name","file_format","ambisonics_order",
                     "sample_rate_hz","num_channels","num_receivers","is_binaural",
                     "duration_s","t60_s","distance_m","room_type","los","file_path"]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        st.dataframe(df[cols], use_container_width=True, hide_index=True, height=520)

        with st.expander("Path check (first 25)", expanded=False):
            if not df.empty:
                sample = df.head(25).to_dict("records")
                ok = miss = out = 0
                rows = []
                for r in sample:
                    rp = _best_local_path(r)
                    if not rp or not Path(rp).exists():
                        miss += 1
                        rows.append(("MISSING", r.get("dataset",""), r.get("file_path","")))
                    else:
                        under_root = (DATA_ROOT in Path(rp).parents or Path(rp) == DATA_ROOT)
                        if under_root or ALLOW_OUTSIDE:
                            ok += 1
                            rows.append(("OK", r.get("dataset",""), str(rp)))
                        else:
                            out += 1
                            rows.append(("OUTSIDE", r.get("dataset",""), str(rp)))
                st.write({"OK": ok, "MISSING": miss, "OUTSIDE": out})
                st.code("\n".join([" | ".join(map(str, r)) for r in rows]))


    # ===== Distributions & Export for latest results =====
    last_df = st.session_state.get("_last_df")
    if last_df is not None and not last_df.empty:
        st.markdown("---")
        st.header("Distributions")
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

        # metric eligibility (how many datasets have non-null for this metric)
        def _eligible(df, col):
            try:
                return int(df.dropna(subset=[col]).groupby("dataset")[col].count().shape[0])
            except Exception:
                return 0


        with st.expander("Save plots", expanded=False):
            fmt_choice = st.selectbox("Export format", ["png","svg","pdf","html","json"], index=0, key="plot_fmt")
            st.caption("PNG/SVG/PDF require the lightweight vl-convert-python backend (installed).")

        def _export_button(chart: alt.Chart, title: str, fname: str, key: str):
            if st.session_state.get("plot_fmt"):
                try:
                    data, mime = chart_bytes(chart, st.session_state["plot_fmt"])
                    st.download_button(
                        f"Save '{title}' as .{st.session_state['plot_fmt']}",
                        data=data, file_name=f"{fname}.{st.session_state['plot_fmt']}",
                        mime=mime, key=key
                    )
                except Exception as e:
                    st.info(f"Chart export: {e}")

        if "volume_log10" in dfm.columns and not dfm["volume_log10"].dropna().empty:
            chart_v = _make_hist_bins_v2(dfm, "volume_log10", "Volume (log10 m³)", topk, facet, bins, strict_topk)
            st.altair_chart(chart_v, use_container_width=True)
            _export_button(chart_v, "Volume (log10 m³)", "volume_log10", "save_vol_chart")

        if "t60_s" in dfm.columns and not dfm["t60_s"].dropna().empty:
            chart_t = _make_hist_bins_v2(dfm, "t60_s", "RT60 (s)", topk, facet, bins, strict_topk)
            st.altair_chart(chart_t, use_container_width=True)
            _export_button(chart_t, "RT60 (s)", "rt60", "save_rt_chart")

        if "bp_log10" in dfm.columns and not dfm["bp_log10"].dropna().empty:
            chart_b = _make_hist_bins_v2(dfm, "bp_log10", "Boundary Points (log10)", topk, facet, bins, strict_topk)
            st.altair_chart(chart_b, use_container_width=True)
            _export_button(chart_b, "Boundary Points (log10)", "boundary_points", "save_bp_chart")

        st.markdown("---")
        st.header("Export (local files)")
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
            if st.button("Build ZIP now", type="secondary", key="zip_build_btn"):
                with st.spinner("Collecting files and building ZIP…"):
                    zbytes, copied, candidates = build_zip_bytes(last_df, max_files=None)
                    st.session_state["zip_bytes"] = zbytes
                    st.session_state["zip_ready"] = True
                    st.session_state["zip_diag"] = {"copied":copied,"candidates":candidates}
                    st.session_state["zip_note"]  = f"Added {copied} of {candidates} candidate files. Includes manifest.csv."
            if st.session_state["zip_ready"] and st.session_state.get("zip_bytes"):
                st.success(st.session_state.get("zip_note","ZIP ready."))
                st.write("Copy summary:", st.session_state.get("zip_diag", {}))
                try:
                    exports_dir = Path(os.environ.get("RIR_EXPORTS","/home/on61ewex/Database/data/_exports"))
                    exports_dir.mkdir(parents=True, exist_ok=True)
                    out_file = exports_dir / "rir_export_page.zip"
                    with open(out_file, "wb") as f:
                        f.write(st.session_state["zip_bytes"])
                    st.caption(f"Saved a copy to: {out_file}")
                except Exception as e:
                    st.info(f"Could not save copy to _exports: {e}")
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
        st.dataframe(distincts_and_ranges(_db_sig())[0], use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()


