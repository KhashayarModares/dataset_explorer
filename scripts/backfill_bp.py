#!/usr/bin/env python3
import re, math, sqlite3, argparse

VOL_RE = re.compile(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)[ ]*m?", re.IGNORECASE)

def ensure_cols(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(rirs)")}
    changed = False
    if "boundary_points" not in cols:
        con.execute("ALTER TABLE rirs ADD COLUMN boundary_points REAL"); changed = True
    if "bp_log10" not in cols:
        con.execute("ALTER TABLE rirs ADD COLUMN bp_log10 REAL"); changed = True
    if "bp_source" not in cols:
        con.execute("ALTER TABLE rirs ADD COLUMN bp_source TEXT"); changed = True
    if changed: con.commit()

def _val(row, key):
    # Safe access for sqlite3.Row
    try:
        return row[key]
    except Exception:
        return None

def parse_volume(room_dims_m):
    if not isinstance(room_dims_m, str): return None
    m = VOL_RE.search(room_dims_m)
    if not m: return None
    try:
        d = float(m.group(1)); w = float(m.group(2)); h = float(m.group(3))
        v = d*w*h
        return v if v > 0 else None
    except Exception:
        return None

def best_bp_for_row(row):
    """Priority: boundary_points -> volume(D*W*H) -> num_receivers -> distance_m"""
    pts = _val(row, "boundary_points")
    try:
        if pts is not None and float(pts) > 0:
            return (math.log10(float(pts)), "points")
    except Exception:
        pass

    vol = parse_volume(_val(row, "room_dims_m"))
    if vol:
        return (math.log10(vol), "proxy_volume")

    nr = _val(row, "num_receivers")
    try:
        if nr is not None and float(nr) > 0:
            return (math.log10(float(nr)), "proxy_receivers")
    except Exception:
        pass

    dm = _val(row, "distance_m")
    try:
        if dm is not None and float(dm) > 0:
            return (math.log10(float(dm)), "proxy_distance")
    except Exception:
        pass

    return (None, None)

def main():
    ap = argparse.ArgumentParser(description="Backfill bp_log10 (Bp proxy) into SQLite.")
    ap.add_argument("--db", required=True, help="SQLite DB path (rir_meta_v3.db)")
    ap.add_argument("--limit", type=int, default=0, help="Limit rows (0 = all)")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    ensure_cols(con)

    sql = "SELECT id, boundary_points, room_dims_m, num_receivers, distance_m FROM rirs"
    if args.limit and args.limit > 0:
        sql += f" LIMIT {int(args.limit)}"
    rows = con.execute(sql).fetchall()

    ok = skip = 0
    for r in rows:
        rid = r["id"]
        bp_log10, src = best_bp_for_row(r)
        if bp_log10 is not None:
            con.execute("UPDATE rirs SET bp_log10=?, bp_source=? WHERE id=?", (float(bp_log10), src, rid))
            ok += 1
        else:
            skip += 1
    con.commit()

    # helpful index for plotting
    con.execute("CREATE INDEX IF NOT EXISTS idx_rirs_bp ON rirs(bp_log10)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_rirs_dataset ON rirs(dataset)")
    con.commit(); con.close()
    print(f"[done] bp_log10 updated OK={ok}, SKIP(no signal)={skip}")

if __name__ == "__main__":
    main()
