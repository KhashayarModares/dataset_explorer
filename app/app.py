#!/usr/bin/env python3
import os, io, zipfile, shutil, tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DB_PATH   = os.environ.get("RIR_DB",        "/home/on61ewex/Database/db/rir_meta_v3.db")
DB_PATH   = os.environ.get("RIR_DB_PATH",   DB_PATH)
DATA_ROOT = Path(os.environ.get("RIR_DATA_ROOT", "/home/on61ewex/Database/data"))
st.set_page_config(page_title="RIR Explorer", layout="wide")

@st.cache_resource
def get_engine()->Engine:
    pgurl=os.environ.get("RIR_PGURL")
    if pgurl: return create_engine(pgurl, pool_pre_ping=True, future=True)
    url=DB_PATH if DB_PATH.startswith(("sqlite:///","postgresql://")) else f"sqlite:///{DB_PATH}"
    eng=create_engine(url, future=True)
    if eng.dialect.name=="sqlite":
        with eng.begin() as con:
            con.exec_driver_sql("PRAGMA journal_mode=WAL;")
            con.exec_driver_sql("PRAGMA synchronous=NORMAL;")
    return eng

def run_df(sql:str, params:Dict[str,Any]=None)->pd.DataFrame:
    with get_engine().begin() as con:
        return pd.read_sql_query(text(sql), con, params=params or {})

def run_scalar(sql:str, params:Dict[str,Any]=None)->Any:
    with get_engine().begin() as con:
        return con.execute(text(sql), params or {}).scalar()

@st.cache_data(ttl=300, show_spinner=False)
def distincts_and_ranges():
    datasets = run_df("SELECT dataset, COUNT(*) AS n FROM rirs GROUP BY dataset ORDER BY dataset")
    def _distinct(col):
        df = run_df(f"SELECT DISTINCT {col} AS v FROM rirs WHERE {col} IS NOT NULL ORDER BY {col}")
        return df["v"].tolist()
    srs   = _distinct("sample_rate_hz")
    rooms = _distinct("room_type")
    fmts  = _distinct("file_format")
    kinds = _distinct("data_kind")
    ambs  = _distinct("ambisonics_order")
    # LOS is only LOS / noLOS after normalization; derive Yes/No for UI
    los_raw = _distinct("los")
    los_ui = []
    if "LOS" in los_raw: los_ui.append("Yes")
    if "noLOS" in los_raw: los_ui.append("No")
    mm = run_df("""
        SELECT MIN(t60_s) AS tmin, MAX(t60_s) AS tmax,
               MIN(distance_m) AS dmin, MAX(distance_m) AS dmax
        FROM rirs
    """).iloc[0].to_dict()
    return datasets, srs, rooms, fmts, kinds, ambs, los_ui, mm

def build_where(filters:Dict[str,Any])->Tuple[str,Dict[str,Any]]:
    w=["1=1"]; p={}
    def _in(col,vals,key):
        if not vals: return
        ph=[]
        for i,v in enumerate(vals):
            k=f"{key}_{i}"; p[k]=v; ph.append(f":{k}")
        w.append(f"{col} IN ({', '.join(ph)})")
    _in("dataset", filters.get("datasets") or [], "ds")
    _in("sample_rate_hz", filters.get("sr") or [], "sr")
    _in("file_format", filters.get("fmts") or [], "fmt")
    _in("data_kind", filters.get("kinds") or [], "kind")
    _in("ambisonics_order", filters.get("ambs") or [], "amb")
    if filters.get("bina") is not None:
        p["bina"]=1 if filters["bina"] else 0
        w.append("is_binaural = :bina")
    if filters.get("room"):
        p["room"]=filters["room"]; w.append("room_type = :room")
    if filters.get("los") is not None:
        # map UI Yes/No to DB 'LOS'/'noLOS'
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

def build_query(filters:Dict[str,Any], limit:int, offset:int)->Tuple[str,Dict[str,Any]]:
    where, p = build_where(filters)
    p["limit"]=int(limit); p["offset"]=int(offset)
    return f"SELECT * FROM rirs WHERE {where} ORDER BY dataset, file_name LIMIT :limit OFFSET :offset", p

def prepare_zip(df: pd.DataFrame) -> bytes:
    staging = Path(tempfile.mkdtemp(prefix="rir_export_"))
    try:
        for _, r in df.iterrows():
            fp = str(r.get("file_path","") or "")
            if not fp: continue
            p = Path(fp)
            if p.exists() and DATA_ROOT in p.resolve().parents:
                dest = staging / p.name
                i=1
                while dest.exists():
                    dest = staging / f"{p.stem}__{i}{p.suffix}"; i+=1
                shutil.copy2(p, dest)
        (staging/"manifest.csv").write_bytes(df.to_csv(index=False).encode("utf-8"))
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_DEFLATED) as z:
            for item in staging.iterdir(): z.write(item, arcname=item.name)
        buf.seek(0); return buf.read()
    finally:
        shutil.rmtree(staging, ignore_errors=True)

def main():
    eng = get_engine()
    st.info(f"Connected to **SQLite** at `{DB_PATH}`" if eng.dialect.name=="sqlite" else "Connected")
    st.title("RIR Explorer")

    datasets, sr_opts, room_opts, fmt_opts, kind_opts, amb_opts, los_opts_ui, mm = distincts_and_ranges()

    with st.sidebar:
        st.header("Filters")
        ds_sel = st.multiselect("Datasets", options=datasets["dataset"].tolist(), default=datasets["dataset"].tolist())
        fmt_sel  = st.multiselect("Format", options=fmt_opts)
        kind_sel = st.multiselect("Data kind", options=kind_opts)
        sr_sel   = st.multiselect("Sample rates (Hz)", options=sr_opts)
        bina     = st.selectbox("Binaural?", ["Any","Yes","No"])
        room     = st.selectbox("Room type", ["Any"] + room_opts)
        los_ui   = st.selectbox("Line of sight", ["Any"] + los_opts_ui)  # Yes/No only
        amb_sel  = st.multiselect("Ambisonics order", options=[v for v in amb_opts if pd.notna(v)])

        # T60 slider only when we have a true range
        t60_range=None; include_null_t60=True
        tmin_db, tmax_db = mm.get("tmin"), mm.get("tmax")
        if pd.notna(tmin_db) and pd.notna(tmax_db) and float(tmin_db) < float(tmax_db):
            t60_range = st.slider("T60 (s)", float(round(float(tmin_db),2)), float(round(float(tmax_db),2)),
                                  (float(round(float(tmin_db),2)), float(round(float(tmax_db),2))), step=0.05)
            include_null_t60 = st.checkbox("Include rows with missing T60", value=True)

        dist_max=None
        if st.checkbox("Apply distance ≤ (m) filter", value=False):
            dmax_db=mm.get("dmax")
            dist_max=st.number_input("Distance ≤ (m)", value=float(dmax_db) if pd.notna(dmax_db) else 0.0,
                                     step=0.1, format="%.2f")

        name_like = st.text_input("Filename contains", "")
        path_like = st.text_input("File path contains", "")
        page_size = st.selectbox("Page size", [1000, 5000, 10000], index=1)
        page_no   = st.number_input("Page (1-based)", min_value=1, value=1, step=1)
        do_search = st.button("Search", type="primary")

    if do_search:
        filters = {
          "datasets": ds_sel,
          "fmts": fmt_sel,
          "kinds": kind_sel,
          "sr": sr_sel,
          "bina": None if bina=="Any" else (bina=="Yes"),
          "room": None if room=="Any" else room,
          "los": None if los_ui=="Any" else los_ui,
          "ambs": amb_sel,
          "t60": t60_range,
          "t60_inc_null": include_null_t60,
          "dist_max": dist_max,
          "name_like": (name_like or "").strip() or None,
          "path_like": (path_like or "").strip() or None,
        }
        count_sql, count_params = build_where(filters)
        total = run_scalar(f"SELECT COUNT(*) FROM rirs WHERE {count_sql}", count_params) or 0
        offset = (int(page_no)-1)*int(page_size)
        sql, params = build_query(filters, limit=page_size, offset=offset)
        df = run_df(sql, params)

        st.success(f"Matched rows: {total:,} | Showing {len(df):,} (page {page_no})")
        with st.expander("Debug (SQL)"): st.code(sql); st.write("Params:", params)

        c1, c2 = st.columns(2)
        if "t60_s" in df.columns and not df["t60_s"].dropna().empty:
            chart = alt.Chart(df.dropna(subset=["t60_s"])).mark_bar().encode(
                x=alt.X('t60_s:Q', bin=alt.Bin(maxbins=40), title='T60 (s)'),
                y=alt.Y('count()', title='Count')
            ).properties(height=240, title="T60 distribution")
            c1.altair_chart(chart, use_container_width=True)
        if "file_format" in df.columns and not df.empty:
            fmt_cnt = df.groupby("file_format", dropna=False)["file_format"].count().reset_index(name="count")
            fmt_bar = alt.Chart(fmt_cnt).mark_bar().encode(
                x=alt.X('count:Q', title='Count'),
                y=alt.Y('file_format:N', sort='-x', title='Format')
            ).properties(height=240, title="Rows by format")
            c2.altair_chart(fmt_bar, use_container_width=True)

        preferred = ["dataset","file_name","file_format","data_kind","ambisonics_order",
                     "sample_rate_hz","num_channels","num_receivers","is_binaural",
                     "duration_s","t60_s","distance_m","room_type","los","file_path"]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        st.dataframe(df[cols], use_container_width=True, hide_index=True, height=520)

        st.download_button("Download CSV (this page)", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="rir_results_page.csv", mime="text/csv")
        if st.button("Prepare ZIP of (this page) files"):
            zip_bytes = prepare_zip(df); st.download_button("Download ZIP", data=zip_bytes,
                                                            file_name="rir_export_page.zip", mime="application/zip")
    with st.expander("Overview"):
        st.write("Datasets and counts:")
        st.dataframe(distincts_and_ranges()[0], use_container_width=True, hide_index=True)
if __name__=="__main__": main()
