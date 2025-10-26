#!/usr/bin/env python3
# tag_invalids.py
#
# Tag remaining INVALIDS quickly by streaming CID-SMILES.gz and selecting only invalid CIDs
# via a DB-built bitmask. Inserts tags in large batches. No per-row SELECTs in the hot loop.
#
# Columns written to invalid_tags:
#   cid TEXT PRIMARY KEY,
#   tags TEXT,           -- comma separated motif names or '-' if none
#   heavy INTEGER,       -- # heavy atoms
#   charge INTEGER,      -- total formal charge
#   rings INTEGER,       -- # rings
#   max_ring INTEGER,    -- largest ring size
#   elements TEXT        -- e.g. "C,H,N,O,P,S"
#
# (Optional) If you want reason copied too, see the post-step below to fill reason via a single join.
#
import argparse, gzip, sys, time, sqlite3, os
from typing import List, Tuple

# ---------- RDKit setup ----------
try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
    from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
except Exception as e:
    print("[error] RDKit required (pip install rdkit-pypi or conda).", file=sys.stderr)
    sys.exit(2)

# ---------- motif library (compile once) ----------
SMARTS = {
    "nitro":           "[N+](=O)[O-]",
    "carboxylate":     "C(=O)[O-]",
    "sulfonate":       "S(=O)(=O)[O-]",
    "phosphate":       "P(=O)(O)[O-]",        # broad P(V) oxyanions
    "quaternary_N":    "[N+](C)(C)C",
    "zwitterion":      "[N+].*[O-]",
    "peroxide":        "O-O",
    "azo":             "N=N",
    "azide":           "N=[N+]=N",
    "diazonium":       "[N+]#N",
    "nitrile":         "C#N",
    "OOO_chain":       "O-O-O",
    "NNN_chain":       "N-N-N",
    "SSS_chain":       "S-S-S",
}
SMARTS_Q = {name: Chem.MolFromSmarts(p) for name, p in SMARTS.items()}

# ---------- SQLite helpers ----------
def open_db_w(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=60.0)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    con.execute("""
        CREATE TABLE IF NOT EXISTS invalid_tags(
            cid       TEXT PRIMARY KEY,
            tags      TEXT NOT NULL,
            heavy     INTEGER,
            charge    INTEGER,
            rings     INTEGER,
            max_ring  INTEGER,
            elements  TEXT
        )
    """)
    # helpful index for later joins
    con.execute("CREATE INDEX IF NOT EXISTS ix_scan_lookup ON scan_results(ruleset, alpha_pi, cid, status)")
    return con

class BitMask:
    def __init__(self, nbits: int):
        self.n = nbits
        self.buf = bytearray((nbits + 7)//8)
    def set(self, i: int):
        if 0 <= i < self.n:
            self.buf[i>>3] |= (1 << (i & 7))
    def test(self, i: int) -> bool:
        if 0 <= i < self.n:
            return (self.buf[i>>3] >> (i & 7)) & 1 == 1
        return False
    def clear(self, i: int):
        if 0 <= i < self.n:
            self.buf[i>>3] &= ~(1 << (i & 7))

def build_invalid_mask(db: str, ruleset: str, alpha: float) -> Tuple[BitMask,int,int]:
    con = sqlite3.connect(f"file:{db}?mode=ro&cache=shared", uri=True, timeout=30.0)
    con.execute("PRAGMA query_only=ON;"); con.execute("PRAGMA read_uncommitted=ON;")
    cur = con.cursor()
    maxcid = cur.execute("""
        SELECT MAX(CAST(cid AS INTEGER)) FROM scan_results
        WHERE status='invalid' AND ruleset=? AND ABS(alpha_pi-?)<1e-12
    """, (ruleset, alpha)).fetchone()[0] or 0
    mask = BitMask(int(maxcid)+1 if maxcid else 1)
    cur.execute("""
        SELECT CAST(cid AS INTEGER) FROM scan_results
        WHERE status='invalid' AND ruleset=? AND ABS(alpha_pi-?)<1e-12
    """, (ruleset, alpha))
    rows = cur.fetchmany(1_000_000)
    n = 0
    while rows:
        for (cid,) in rows:
            if cid is not None:
                mask.set(int(cid)); n += 1
        rows = cur.fetchmany(1_000_000)
    cur.close(); con.close()
    return mask, n, maxcid

# ---------- tagging ----------
def tag_features(smi: str) -> Tuple[str, int, int, int, int, str]:
    """Return (tags_csv, heavy, charge, rings, max_ring, elements_csv) for a SMILES."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return "parse_fail", 0, 0, 0, 0, ""
    # features
    charge = sum(a.GetFormalCharge() for a in m.GetAtoms())
    heavy = sum(1 for a in m.GetAtoms() if a.GetAtomicNum()!=1)
    ri = m.GetRingInfo()
    rings = ri.NumRings()
    try:
        sssr = Chem.GetSymmSSSR(m)
        max_ring = max((len(list(r)) for r in sssr), default=0)
    except Exception:
        max_ring = 0
    elements = ",".join(sorted({a.GetSymbol() for a in m.GetAtoms()}))
    # motifs
    tags = []
    for name, q in SMARTS_Q.items():
        if q and m.HasSubstructMatch(q):
            tags.append(name)
    tags_csv = ",".join(tags) if tags else "-"
    return tags_csv, heavy, charge, rings, max_ring, elements

def main():
    ap = argparse.ArgumentParser(description="Fast tagger: stream gz, tag only invalid CIDs via DB mask, batch-insert tags.")
    ap.add_argument("--cid-smiles-gz", required=True)
    ap.add_argument("--sqlite-out", required=True)
    ap.add_argument("--ruleset", default="RS3")
    ap.add_argument("--alpha-pi", type=float, default=1.15)
    ap.add_argument("--batch", type=int, default=100000)
    ap.add_argument("--progress-every", type=int, default=1000000)
    ap.add_argument("--stdin", action="store_true", help="read SMILES from stdin (use with: pigz -dc CID-SMILES.gz | ...)")
    args = ap.parse_args()

    print("[mask] building invalid CID mask …", flush=True)
    mask, ninv, maxcid = build_invalid_mask(args.sqlite_out, args.ruleset, args.alpha_pi)
    print(f"[mask] invalid rows: {ninv}  (max CID ~{maxcid})", flush=True)

    con = open_db_w(args.sqlite_out)
    ins = con.cursor()
    insert_sql = """
        INSERT OR REPLACE INTO invalid_tags
        (cid, tags, heavy, charge, rings, max_ring, elements)
        VALUES (?,?,?,?,?,?,?)
    """

    t0 = time.time()
    total_seen = 0
    touched = 0
    to_insert = []

    def flush():
        nonlocal to_insert
        if not to_insert: return
        ins.executemany(insert_sql, to_insert)
        con.commit()
        to_insert.clear()

    # choose input stream
    if args.stdin:
        fh = sys.stdin
        print("[stream] reading from STDIN", flush=True)
    else:
        fh = gzip.open(args.cid_smiles_gz, "rt", encoding="utf-8", errors="ignore")
        print(f"[stream] reading {args.cid_smiles_gz}", flush=True)

    with fh:
        for line in fh:
            total_seen += 1
            try:
                cid_str, smi = line.rstrip("\n").split("\t", 1)
                cid = int(cid_str)
            except Exception:
                continue
            if not mask.test(cid):
                continue
            # tag
            tags_csv, heavy, charge, rings, max_ring, elements = tag_features(smi)
            to_insert.append((cid_str, tags_csv, heavy, charge, rings, max_ring, elements))
            touched += 1

            if len(to_insert) >= args.batch:
                flush()
                dt = time.time() - t0
                rate = touched/dt if dt>0 else 0.0
                print(f"[commit] touched={touched:,} rate={rate:.1f}/s", flush=True)

            if args.progress_every and (total_seen % args.progress_every == 0):
                dt = time.time() - t0
                rate = touched/dt if dt>0 else 0.0
                print(f"[progress] scanned_gz={total_seen:,} touched={touched:,} rate={rate:.1f}/s", flush=True)

    flush()
    dt = time.time() - t0
    rate = touched/dt if dt>0 else 0.0
    print(f"[done] scanned_gz={total_seen:,} touched={touched:,} avg_rate={rate:.1f}/s db={args.sqlite_out}", flush=True)

if __name__ == "__main__":
    main()