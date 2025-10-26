#!/usr/bin/env python3
# check_pubchem_exists_stream.py
#
# Stream-safe PubChem checker:
#  - Writes to CSV *every batch* (atomic append)
#  - Auto-resume from existing --out (uses query_key if available, else smiles)
#  - SQLite cache + polite backoff + adaptive batch sizes
#
# Usage:
#   python3 check_pubchem_exists_stream.py \
#     --input novel_candidates_stable.csv \
#     --out novel_candidates_pubchem_checked.csv
#
import argparse, os, sys, time, json, random, sqlite3, csv, tempfile
from typing import Dict, List, Tuple, Optional
from urllib.parse import quote
import requests
import pandas as pd

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# ---------------- RDKit helpers (optional) ----------------
def rdkit_enabled():
    try:
        from rdkit import Chem  # noqa
        return True
    except Exception:
        return False

def to_canon_and_inchikey(smi: str) -> Tuple[str, str]:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if not m:
            return smi, ""
        can = Chem.MolToSmiles(m, canonical=True)
        ik = Chem.InchiToInchiKey(Chem.MolToInchi(m))
        return can, ik
    except Exception:
        return smi, ""

# ---------------- SQLite cache ----------------
def open_cache(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30.0)
    con.execute("""CREATE TABLE IF NOT EXISTS smiles_cache(
        key TEXT PRIMARY KEY,
        query_smiles TEXT,
        inchikey TEXT,
        cids TEXT,
        canonical_smiles TEXT,
        iupac_name TEXT,
        ts INTEGER
    )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON smiles_cache(ts)")
    return con

def cache_get(con: sqlite3.Connection, key: str) -> Optional[Dict]:
    row = con.execute("SELECT query_smiles, inchikey, cids, canonical_smiles, iupac_name FROM smiles_cache WHERE key=?",(key,)).fetchone()
    if not row: return None
    return {"query_smiles": row[0], "inchikey": row[1] or "", "cids": row[2] or "", "canonical_smiles": row[3] or "", "iupac_name": row[4] or ""}

def cache_put(con: sqlite3.Connection, key: str, query_smiles: str, inchikey: str, cids_csv: str, canonical_smiles: str, iupac: str):
    con.execute("INSERT OR REPLACE INTO smiles_cache(key,query_smiles,inchikey,cids,canonical_smiles,iupac_name,ts) VALUES (?,?,?,?,?,?,?)",
                (key, query_smiles, inchikey, cids_csv, canonical_smiles, iupac, int(time.time())))
    con.commit()

# ---------------- HTTP with backoff ----------------
def polite_get(url: str, timeout: int, max_retries: int, base_sleep: float, quiet: bool) -> Optional[requests.Response]:
    wait = base_sleep
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
        except Exception:
            r = None
        if r is not None and r.status_code == 200:
            return r
        retry_after = 0.0
        status = r.status_code if r is not None else 'ERR'
        if r is not None and r.status_code in (429, 503):
            ra = r.headers.get("Retry-After")
            if ra:
                try: retry_after = float(ra)
                except: retry_after = 0.0
        sleep_for = max(wait, retry_after) + random.uniform(0.05, 0.35)
        if not quiet:
            print(f"[backoff] {status} → sleep {sleep_for:.2f}s (attempt {attempt+1}/{max_retries})", flush=True)
        time.sleep(sleep_for)
        wait = min(wait * 1.8, 15.0)
    return None

def polite_post(url: str, data: List[Tuple[str,str]], timeout: int, max_retries: int, base_sleep: float, quiet: bool) -> Optional[requests.Response]:
    wait = base_sleep
    for attempt in range(max_retries):
        try:
            r = requests.post(url, data=data, timeout=timeout)
        except Exception:
            r = None
        if r is not None and r.status_code == 200:
            return r
        retry_after = 0.0
        status = r.status_code if r is not None else 'ERR'
        if r is not None and r.status_code in (429, 503):
            ra = r.headers.get("Retry-After")
            if ra:
                try: retry_after = float(ra)
                except: retry_after = 0.0
        sleep_for = max(wait, retry_after) + random.uniform(0.05, 0.35)
        if not quiet:
            print(f"[backoff] {status} → sleep {sleep_for:.2f}s (attempt {attempt+1}/{max_retries})", flush=True)
        time.sleep(sleep_for)
        wait = min(wait * 1.8, 15.0)
    return None

# ---------------- PubChem calls ----------------
def fetch_cids_by_smiles_batch(smiles: List[str], timeout: int, retries: int, sleep: float, quiet: bool) -> Dict[str, List[int]]:
    url = f"{PUG}/compound/smiles/cids/JSON"
    post_data = [('smiles', s) for s in smiles]
    r = polite_post(url, post_data, timeout, retries, sleep, quiet)
    result = {s: [] for s in smiles}
    if r is None: return result
    try:
        j = r.json()
        info = j.get("InformationList", {}).get("Information", [])
        for item in info:
            inp = item.get("Input", {})
            s_in = inp.get("smiles")
            cids = item.get("CID", [])
            if s_in in result:
                result[s_in] = list(map(int, cids)) if isinstance(cids, list) else []
    except Exception:
        pass
    return result

def fetch_cids_by_smiles_single(smi: str, timeout: int, retries: int, sleep: float, quiet: bool) -> List[int]:
    url = f"{PUG}/compound/smiles/{quote(smi, safe='')}/cids/JSON"
    r = polite_get(url, timeout, retries, sleep, quiet)
    if r is None: return []
    try:
        j = r.json()
        return list(map(int, j.get("IdentifierList", {}).get("CID", [])))
    except Exception:
        return []

def fetch_cids_by_inchikey(ik: str, timeout: int, retries: int, sleep: float, quiet: bool) -> List[int]:
    url = f"{PUG}/compound/inchikey/{quote(ik, safe='')}/cids/JSON"
    r = polite_get(url, timeout, retries, sleep, quiet)
    if r is None: return []
    try:
        j = r.json()
        return list(map(int, j.get("IdentifierList", {}).get("CID", [])))
    except Exception:
        return []

def fetch_props_for_cids(cids: List[int], timeout: int, retries: int, sleep: float, quiet: bool) -> Dict[int, Dict[str,str]]:
    props = "CanonicalSMILES,IUPACName"
    out: Dict[int, Dict[str,str]] = {}
    step = 100
    for i in range(0, len(cids), step):
        chunk = cids[i:i+step]
        cid_str = ",".join(map(str, chunk))
        url = f"{PUG}/compound/cid/{cid_str}/property/{props}/JSON"
        r = polite_get(url, timeout, retries, sleep, quiet)
        if r is None: continue
        try:
            j = r.json()
            for row in j.get("PropertyTable", {}).get("Properties", []):
                out[int(row["CID"])] = {
                    "CanonicalSMILES": row.get("CanonicalSMILES",""),
                    "IUPACName": row.get("IUPACName","")
                }
        except Exception:
            pass
        time.sleep(sleep)
    return out

# ---------------- Streaming CSV append ----------------
OUT_COLUMNS = None  # filled at runtime

def atomic_append_rows(out_path: str, rows: List[Dict[str, object]], header_fields: List[str]):
    """Append rows atomically: write to temp file then os.replace() into an '.append' file, then append to main."""
    # If main file doesn't exist, write header first.
    new_file = not os.path.exists(out_path)
    # We append directly to out_path safely: write to temp file then append its contents
    with tempfile.NamedTemporaryFile("w", delete=False, newline="", encoding="utf-8") as tf:
        writer = csv.DictWriter(tf, fieldnames=header_fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
        tmpname = tf.name
    # Append temp to out_path
    with open(tmpname, "r", encoding="utf-8") as src, open(out_path, "a", newline="", encoding="utf-8") as dst:
        if new_file:
            # temp already includes header; out is empty; write full temp
            dst.write(src.read())
        else:
            # skip header line on append
            first = True
            for line in src:
                if first:
                    first = False
                    # skip header line from tmp
                    continue
                dst.write(line)
    os.remove(tmpname)

def load_processed_keys(out_path: str) -> set:
    """Return set of processed query keys from an existing out CSV."""
    if not os.path.exists(out_path): return set()
    try:
        # load minimally to avoid big memory: read 'query_key' if present else 'smiles'
        processed = set()
        with open(out_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            use_col = "query_key" if "query_key" in reader.fieldnames else ("smiles" if "smiles" in reader.fieldnames else None)
            if use_col is None: return set()
            for row in reader:
                key = (row.get(use_col) or "").strip()
                if key:
                    processed.add(key)
        return processed
    except Exception:
        return set()

# ---------------- CLI & main ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Stream-safe, resumable PubChem checker.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--resume-file", default="pc_resume.json")  # optional now
    ap.add_argument("--cache-db", default="pubchem_cache.db")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--min-batch", type=int, default=10)
    ap.add_argument("--max-batch", type=int, default=200)
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=40)
    ap.add_argument("--sleep", type=float, default=0.30)
    ap.add_argument("--normalize-smiles", action="store_true")
    ap.add_argument("--use-inchikey", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()

def save_resume(path: str, cursor: int, batch_size: int):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"cursor": cursor, "batch_size": batch_size, "ts": time.time()}, f)
    os.replace(tmp, path)

def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"[error] input not found: {args.input}", file=sys.stderr); sys.exit(2)

    df = pd.read_csv(args.input)
    if args.smiles_col not in df.columns:
        print(f"[error] column '{args.smiles_col}' not in CSV (have {list(df.columns)})", file=sys.stderr); sys.exit(2)

    # keep stable only if present
    if "stable" in df.columns:
        df = df[df["stable"] == True].reset_index(drop=True)

    have_rdkit = rdkit_enabled()
    if (args.normalize_smiles or args.use_inchikey) and not have_rdkit:
        print("[warn] RDKit not available; disabling --normalize-smiles/--use-inchikey")
        args.normalize_smiles = False
        args.use_inchikey = False

    # Build record list (index, original smiles, query_key, inchikey)
    records = []
    for i in range(len(df)):
        smi = str(df.at[i, args.smiles_col])
        if not smi or smi.lower() == "nan": continue
        key_smi = smi
        ik = ""
        if args.normalize_smiles or args.use_inchikey:
            can, ik = to_canon_and_inchikey(smi)
            if args.normalize_smiles and can:
                key_smi = can
        key = ik if (args.use_inchikey and ik) else key_smi
        records.append((i, smi, key, ik))

    # Resume from existing OUT: skip anything already written
    processed_keys = load_processed_keys(args.out)
    pending = [(i,smi,key,ik) for (i,smi,key,ik) in records if key not in processed_keys]

    if not args.quiet:
        print(f"[info] total rows={len(df)}  to_process={len(pending)}  already_done={len(records)-len(pending)}")

    cache = open_cache(args.cache_db)

    # Output header fields: original columns + our annotations + query_key/inchikey
    global OUT_COLUMNS
    OUT_COLUMNS = list(df.columns) + [
        "pubchem_found", "pubchem_cids", "pubchem_first_cid",
        "pubchem_canonical_smiles", "pubchem_iupac_name",
        "query_key", "inchikey"
    ]

    batch = args.batch_size
    cursor = 0

    # try resume file for batch size/cursor, but clamp by pending
    if os.path.exists(args.resume_file):
        try:
            r = json.loads(open(args.resume_file).read())
            batch = max(args.min_batch, min(args.max_batch, int(r.get("batch_size", batch))))
        except Exception:
            pass

    while cursor < len(pending):
        # form a batch by *query key* uniqueness
        keys = []
        idxs = []
        # gather up to 'batch' unique keys, preserving order
        j = cursor
        seen_in_batch = set()
        while j < len(pending) and len(keys) < batch:
            i, smi, key, ik = pending[j]
            if key not in seen_in_batch:
                keys.append(key)
                seen_in_batch.add(key)
            idxs.append(j)
            j += 1

        # cache lookups + list of unknown keys
        key_to_result = {}
        unknown_keys = []
        for k in keys:
            c = cache_get(cache, k)
            if c:
                key_to_result[k] = c
            else:
                unknown_keys.append(k)

        # query unknown keys
        if unknown_keys:
            # Decide: inchikey single GET or smiles batch
            if args.use_inchikey:
                for k in unknown_keys:
                    if len(k) == 27 and k.count('-') == 2:
                        cids = fetch_cids_by_inchikey(k, args.timeout, args.max_retries, args.sleep, args.quiet)
                    else:
                        cids = fetch_cids_by_smiles_single(k, args.timeout, args.max_retries, args.sleep, args.quiet)
                    key_to_result[k] = {"query_smiles": "", "inchikey": k if len(k)==27 else "", "cids": ",".join(map(str,cids)),
                                        "canonical_smiles": "", "iupac_name": ""}
                    time.sleep(args.sleep)
            else:
                # batch by smiles
                got = fetch_cids_by_smiles_batch(unknown_keys, args.timeout, args.max_retries, args.sleep, args.quiet)
                # fetch properties for all found CIDs
                all_cids = sorted({cid for cids in got.values() for cid in cids})
                props = fetch_props_for_cids(all_cids, args.timeout, args.max_retries, args.sleep, args.quiet)
                for k in unknown_keys:
                    cids = got.get(k, []) or []
                    cids_csv = ",".join(map(str, cids)) if cids else ""
                    can = props.get(cids[0], {}).get("CanonicalSMILES","") if cids else ""
                    iupac = props.get(cids[0], {}).get("IUPACName","") if cids else ""
                    key_to_result[k] = {"query_smiles": "", "inchikey": "", "cids": cids_csv,
                                        "canonical_smiles": can, "iupac_name": iupac}
                    time.sleep(args.sleep)

            # cache all unknowns we just resolved
            for k in unknown_keys:
                ent = key_to_result[k]
                cache_put(cache, k, ent.get("query_smiles",""), ent.get("inchikey",""),
                          ent.get("cids",""), ent.get("canonical_smiles",""), ent.get("iupac_name",""))

        # build output rows for this batch and append immediately
        out_rows: List[Dict[str, object]] = []
        for j in range(cursor, min(j, len(pending))):
            i, smi, key, ik = pending[j]
            base = {col: df.at[i, col] for col in df.columns}
            ent = key_to_result.get(key, {"cids":"", "canonical_smiles":"", "iupac_name":""})
            cids_csv = ent.get("cids","")
            cids_list = [int(x) for x in cids_csv.split(",")] if cids_csv else []
            row = dict(base)
            row.update({
                "pubchem_found": bool(cids_list),
                "pubchem_cids": cids_csv,
                "pubchem_first_cid": cids_list[0] if cids_list else None,
                "pubchem_canonical_smiles": ent.get("canonical_smiles",""),
                "pubchem_iupac_name": ent.get("iupac_name",""),
                "query_key": key,
                "inchikey": ik
            })
            out_rows.append(row)

        # append to CSV (atomic per batch)
        atomic_append_rows(args.out, out_rows, OUT_COLUMNS)

        # advance cursor
        cursor = j

        # adapt batch a touch upward if we’re cruising
        if batch < args.max_batch:
            batch = min(args.max_batch, int(batch * 1.15))

        save_resume(args.resume_file, cursor, batch)

        print(f"[batch] wrote {len(out_rows)} rows → {args.out}  progress={cursor}/{len(pending)}  next_batch={batch}", flush=True)
        time.sleep(args.sleep + random.uniform(0.02, 0.08))

    print(f"[done] all pending processed. output: {args.out}")

if __name__ == "__main__":
    main()