#!/usr/bin/env python3
# check_pubchem_exists.py
#
# Stream-safe checker using a local PubChem SMILES dump
#  - Writes to CSV *every batch* (atomic append)
#  - Auto-resume from existing --out (uses query_key if available, else smiles)
#  - Builds/uses a local SQLite index from CID-SMILES.gz (no network)
#
# Example:
#   python3 check_pubchem_exists.py \
#     --input novel_candidates_stable.csv \
#     --out novel_candidates_pubchem_checked.csv \
#     --cid-smiles /mnt/sdc1/chemicals/pubchem/Extras/CID-SMILES.gz \
#     --smiles-index /mnt/sdc1/chemicals/pubchem/Extras/pubchem_smiles.sqlite
#
import argparse, os, sys, time, json, random, sqlite3, csv, tempfile, gzip
from typing import Dict, List, Tuple, Optional
import pandas as pd

# ---------------- RDKit helpers (optional; not used unless --normalize-smiles) ----------------
def rdkit_enabled():
    try:
        from rdkit import Chem  # noqa
        return True
    except Exception:
        return False

def to_canonical(smi: str) -> str:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if not m:
            return smi
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return smi

# ---------------- SQLite: local SMILES index ----------------
def open_index(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=120.0)
    con.execute("""
        CREATE TABLE IF NOT EXISTS smiles_index (
            smi TEXT PRIMARY KEY,
            cid TEXT
        )
    """)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    con.execute("CREATE INDEX IF NOT EXISTS idx_smi ON smiles_index(smi)")
    return con

def index_rowcount(con: sqlite3.Connection) -> int:
    try:
        return int(con.execute("SELECT COUNT(*) FROM smiles_index").fetchone()[0])
    except Exception:
        return 0

def ensure_smiles_index(gz_path: str, db_path: str, *, canonicalize: bool = False):
    if not os.path.exists(gz_path):
        raise SystemExit(f"[error] CID-SMILES.gz not found: {gz_path}")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = open_index(db_path)
    if index_rowcount(con) > 0:
        con.close()
        return  # already built
    print(f"[build] indexing {gz_path} → {db_path} (canonicalize={canonicalize})", file=sys.stderr)
    have_rd = rdkit_enabled() if canonicalize else False
    cur = con.cursor()
    batch = []
    BATCH_SIZE = 100000
    n = 0
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                cid, smi = line.split("\t")
            except ValueError:
                continue
            smi_key = to_canonical(smi) if have_rd else smi
            batch.append((smi_key, cid))
            n += 1
            if len(batch) >= BATCH_SIZE:
                cur.executemany("INSERT OR IGNORE INTO smiles_index (smi, cid) VALUES (?,?)", batch)
                con.commit()
                print(f"[build] indexed {n:,}", file=sys.stderr)
                batch.clear()
    if batch:
        cur.executemany("INSERT OR IGNORE INTO smiles_index (smi, cid) VALUES (?,?)", batch)
        con.commit()
    cur.close()
    print(f"[build] done. total rows={index_rowcount(con):,}", file=sys.stderr)
    con.close()

def lookup_smiles(con: sqlite3.Connection, smi: str) -> Optional[str]:
    row = con.execute("SELECT cid FROM smiles_index WHERE smi=? LIMIT 1", (smi,)).fetchone()
    return row[0] if row else None

# ---------------- Streaming CSV append ----------------
OUT_COLUMNS = None  # set at runtime

def atomic_append_rows(out_path: str, rows: List[Dict[str, object]], header_fields: List[str]):
    new_file = not os.path.exists(out_path)
    with tempfile.NamedTemporaryFile("w", delete=False, newline="", encoding="utf-8") as tf:
        writer = csv.DictWriter(tf, fieldnames=header_fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
        tmpname = tf.name
    with open(tmpname, "r", encoding="utf-8") as src, open(out_path, "a", newline="", encoding="utf-8") as dst:
        if new_file:
            dst.write(src.read())
        else:
            first = True
            for line in src:
                if first:
                    first = False
                    continue
                dst.write(line)
    os.remove(tmpname)

def load_processed_keys(out_path: str) -> set:
    if not os.path.exists(out_path):
        return set()
    try:
        processed = set()
        with open(out_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            use_col = "query_key" if "query_key" in reader.fieldnames else ("smiles" if "smiles" in reader.fieldnames else None)
            if use_col is None:
                return set()
            for row in reader:
                key = (row.get(use_col) or "").strip()
                if key:
                    processed.add(key)
        return processed
    except Exception:
        return set()

# ---------------- CLI & main ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Stream-safe, local PubChem SMILES existence checker (no network).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--resume-file", default="pc_resume.json")
    ap.add_argument("--cache-db", default="pubchem_cache.db")  # kept for compatibility; unused in local-only flow
    # local dump/index
    ap.add_argument("--cid-smiles", required=True, help="Path to CID-SMILES.gz (e.g., /mnt/sdc1/chemicals/pubchem/Extras/CID-SMILES.gz)")
    ap.add_argument("--smiles-index", required=True, help="Path to SQLite index (will be created on first run)")
    ap.add_argument("--canonicalize-local", action="store_true", help="Canonicalize BOTH index keys and input SMILES (uses RDKit; no InChI).")
    # normalization for inputs only (optional)
    ap.add_argument("--normalize-smiles", action="store_true", help="Canonicalize input SMILES before lookup (RDKit; no InChI).")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()

def save_resume(path: str, cursor: int):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"cursor": cursor, "ts": time.time()}, f)
    os.replace(tmp, path)

def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"[error] input not found: {args.input}", file=sys.stderr); sys.exit(2)

    # Build/open local index
    ensure_smiles_index(args.cid_smiles, args.smiles_index, canonicalize=args.canonicalize_local)
    idx_con = open_index(args.smiles_index)

    # Load candidates
    df = pd.read_csv(args.input)
    if args.smiles_col not in df.columns:
        print(f"[error] column '{args.smiles_col}' not in CSV (have {list(df.columns)})", file=sys.stderr); sys.exit(2)
    if "stable" in df.columns:
        df = df[df["stable"] == True].reset_index(drop=True)

    have_rd = rdkit_enabled()
    if (args.normalize_smiles or args.canonicalize_local) and not have_rd:
        print("[warn] RDKit not available; disabling canonicalization flags", file=sys.stderr)
        args.normalize_smiles = False

    # Build record list (index, original smiles, query_key)
    def key_of(s: str) -> str:
        if args.normalize_smiles:
            return to_canonical(s)
        return s

    records = []
    for i in range(len(df)):
        smi = str(df.at[i, args.smiles_col])
        if not smi or smi.lower() == "nan":
            continue
        key = key_of(smi)
        records.append((i, smi, key))

    # Resume from existing OUT: skip anything already written
    processed_keys = load_processed_keys(args.out)
    pending = [(i, smi, key) for (i, smi, key) in records if key not in processed_keys]

    if not args.quiet:
        print(f"[info] total rows={len(df)}  to_process={len(pending)}  already_done={len(records)-len(pending)}", file=sys.stderr)

    # Output header fields: original columns + our annotations + query_key
    global OUT_COLUMNS
    OUT_COLUMNS = list(df.columns) + [
        "pubchem_found", "pubchem_first_cid", "query_key"
    ]

    cursor = 0
    BATCH = 2000  # local lookups are fast; write every ~BATCH rows

    while cursor < len(pending):
        end = min(cursor + BATCH, len(pending))
        out_rows: List[Dict[str, object]] = []

        for j in range(cursor, end):
            i, smi, key = pending[j]
            base = {col: df.at[i, col] for col in df.columns}
            cid = lookup_smiles(idx_con, key)
            row = dict(base)
            row.update({
                "pubchem_found": bool(cid),
                "pubchem_first_cid": cid if cid else None,
                "query_key": key
            })
            out_rows.append(row)

        atomic_append_rows(args.out, out_rows, OUT_COLUMNS)
        cursor = end
        save_resume(args.resume_file, cursor)
        if not args.quiet:
            print(f"[batch] wrote {len(out_rows)} rows → {args.out}  progress={cursor}/{len(pending)}", flush=True)

    idx_con.close()
    print(f"[done] all pending processed. output: {args.out}")

if __name__ == "__main__":
    main()