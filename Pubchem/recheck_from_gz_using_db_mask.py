#!/usr/bin/env python3
# recheck_from_gz_using_db_mask.py
#
# Recheck-only pass using CID-SMILES.gz as the fast stream, using a DB-built mask
# to select which CIDs (current 'invalid') to recompute. Updates DB in batches.

import argparse, gzip, sqlite3, sys, time
from typing import Dict, Tuple, List

# --- import your verifier (source of truth) ---
import molecule_sql_verify as msv

# --- RDKit charged-motif whitelist (short-circuit to salt_accept) ---
from rdkit import Chem

CHARGED_WHITELIST = [
    "[N+](=O)[O-]",      # nitro
    "C(=O)[O-]",         # carboxylate
    "S(=O)(=O)[O-]",     # sulfonate
    "P(=O)(O)[O-]",      # phosphate/phosphonate
    "[N+](C)(C)C",       # simple quaternary ammonium
    "[N+].*[O-]"         # generic betaine / zwitterion
]
_WHITELIST_QUERIES = [Chem.MolFromSmarts(s) for s in CHARGED_WHITELIST]

def _matches_charged_whitelist(smi: str) -> bool:
    m = Chem.MolFromSmiles(smi)
    if not m: return False
    return any(q and m.HasSubstructMatch(q) for q in _WHITELIST_QUERIES)

# --- RDKit helpers for SMILES→(atoms,mult) ---
def atoms_mult_from_smiles(smi: str) -> Tuple[List[str], Dict[Tuple[int,int], int]]:
    from rdkit.Chem import rdchem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError("parse_fail")
    m2 = Chem.Mol(m)
    try:
        Chem.Kekulize(m2, clearAromaticFlags=True)
    except Exception:
        m2 = m
    heavy = [a.GetIdx() for a in m2.GetAtoms() if a.GetAtomicNum() != 1]
    if not heavy:
        raise ValueError("no_heavy")
    remap = {old:i for i,old in enumerate(heavy)}
    H = sum(int(a.GetTotalNumHs()) for a in m2.GetAtoms() if a.GetAtomicNum()!=1)
    atoms = [m2.GetAtomWithIdx(i).GetSymbol() for i in heavy] + ["H"]*H
    mult: Dict[Tuple[int,int], int] = {}
    for b in m2.GetBonds():
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in remap and j in remap:
            a,b2 = sorted((remap[i], remap[j]))
            bt = b.GetBondType()
            v = 1 if bt==rdchem.BondType.SINGLE else 2 if bt==rdchem.BondType.DOUBLE else 3 if bt==rdchem.BondType.TRIPLE else 1
            if v > mult.get((a,b2),0): mult[(a,b2)] = v
    return atoms, mult

def positive_charge_units(smi: str) -> int:
    m = Chem.MolFromSmiles(smi)
    if m is None: return 0
    s = 0
    for a in m.GetAtoms():
        if a.GetAtomicNum()!=1 and a.GetFormalCharge()>0:
            s += a.GetFormalCharge()
    return s

def prepare_atoms_like_db(atoms: List[str], q_units: int) -> List[str]:
    """Mirror DB SALT_Q + parity H-balance (no metal/halo stripping in opposite-way)."""
    from collections import defaultdict
    def counts(lst):
        c=defaultdict(int)
        for a in lst: c[a]+=1
        return c
    msv.SALT_Q = q_units
    atoms1 = msv._hydrogen_balance_fix(atoms, q_units, drop_halos_for_H=False,
                                       salt_delta_metal=0, cation_charge=q_units)
    C = counts(atoms1)
    vmap = msv.assign_valences_heavy(atoms1)
    v_sum = sum(vmap[i] for i in vmap)
    nh = sum(1 for a in atoms1 if a!="H")
    dbe = msv.dbe_from_counts(C)
    H_need = int(v_sum - 2*dbe - 2*(nh - 1))
    H_has  = C.get("H",0)
    delta = H_need - H_has
    if delta!=0:
        if abs(delta)%2==1:
            msv.SALT_Q += (-1 if delta>0 else +1)
        else:
            steps = min(abs(delta)//2, 16)
            msv.SALT_Q += (-2 if delta>0 else +2)*steps
        atoms1 = msv._hydrogen_balance_fix(atoms1, msv.SALT_Q, drop_halos_for_H=False,
                                           salt_delta_metal=0, cation_charge=q_units)
    return atoms1

def recompute_status(smi: str) -> Tuple[str,str]:
    # short-circuit: treat common charged motifs as salts in opposite-way path
    if _matches_charged_whitelist(smi):
        return ("salt_accept", "charged_motif")

    try:
        atoms, mult = atoms_mult_from_smiles(smi)
    except ValueError as e:
        msg=str(e)
        if msg=="parse_fail": return ("parse_fail","parse_fail")
        if msg=="no_heavy":   return ("no_heavy","no_heavy")
        return ("kekulize_fail","kekulize_fail")

    q = positive_charge_units(smi)
    atoms = prepare_atoms_like_db(atoms, q)
    others = [a for a in atoms if a not in msv.BASE_VALENCE]
    if others:
        U=set(others); INORG=getattr(msv,"INORG_ACCEPT",set())
        if U.issubset(INORG): return ("salt_accept","inorg_whitelist")
        else:                  return ("unsupported","unsupported:"+",".join(sorted(U)))
    ok, why = msv.verify_cert(atoms, mult)
    if ok: return ("valid","")
    else:  return ("invalid", why or "verify_fail")

# --- bitset mask for invalid CIDs ---
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
    n=0
    while rows:
        for (cid,) in rows:
            if cid is not None:
                mask.set(int(cid)); n += 1
        rows = cur.fetchmany(1_000_000)
    cur.close(); con.close()
    return mask, n, maxcid

def open_db_w(db: str) -> sqlite3.Connection:
    con = sqlite3.connect(db, timeout=60.0)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    return con

def main():
    ap = argparse.ArgumentParser(description="Fast recheck: stream gz, mask invalid CIDs from DB, update in batches.")
    ap.add_argument("--cid-smiles-gz", required=True)
    ap.add_argument("--sqlite-out", required=True)
    ap.add_argument("--ruleset", default="RS3")
    ap.add_argument("--alpha-pi", type=float, default=1.15)
    ap.add_argument("--batch", type=int, default=50000)
    ap.add_argument("--progress-every", type=int, default=500000)  # gz lines
    args = ap.parse_args()

    # ensure RDKit present
    try:
        from rdkit import Chem  # noqa
    except Exception:
        print("[error] RDKit required (pip install rdkit-pypi or conda).", file=sys.stderr)
        sys.exit(2)

    print(f"[mask] building invalid CID mask from DB…", flush=True)
    mask, ninv, maxcid = build_invalid_mask(args.sqlite_out, args.ruleset, args.alpha_pI if hasattr(args,'alpha_pI') else args.alpha_pi)
    print(f"[mask] invalid rows: {ninv}  (max CID ~{maxcid})", flush=True)

    con = open_db_w(args.sqlite_out)
    upd = con.cursor()
    ts_now = int(time.time())
    to_update = []
    total_seen = 0
    touched = 0
    flipped = 0
    t0 = time.time()

    with gzip.open(args.cid_smiles_gz, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            total_seen += 1
            try:
                cid_str, smi = line.rstrip("\n").split("\t", 1)
                cid = int(cid_str)
            except Exception:
                continue
            if not mask.test(cid):
                continue

            new_status, new_reason = recompute_status(smi)

            if new_status != "invalid":
                flipped += 1
                to_update.append((new_status, new_reason, ts_now, str(cid), args.ruleset, args.alpha_pi))
                mask.clear(cid)
            touched += 1

            if len(to_update) >= args.batch:
                upd.executemany(
                    "UPDATE scan_results SET status=?, reason=?, ts=? WHERE cid=? AND ruleset=? AND alpha_pi=? AND status='invalid'",
                    to_update
                )
                con.commit()
                to_update.clear()
                dt = time.time() - t0
                rate = touched/dt if dt>0 else 0.0
                print(f"[commit] touched={touched} flipped={flipped} rate={rate:.1f}/s", flush=True)

            if args.progress_every and (total_seen % args.progress_every == 0):
                dt = time.time() - t0
                rate = touched/dt if dt>0 else 0.0
                print(f"[progress] scanned_gz={total_seen} touched={touched} flipped={flipped} rate={rate:.1f}/s", flush=True)

    if to_update:
        upd.executemany(
            "UPDATE scan_results SET status=?, reason=?, ts=? WHERE cid=? AND ruleset=? AND alpha_pi=? AND status='invalid'",
            to_update
        )
        con.commit()
        to_update.clear()

    upd.close(); con.close()
    dt = time.time() - t0
    rate = touched/dt if dt>0 else 0.0
    print(f"[done] scanned_gz={total_seen} touched={touched} flipped={flipped} avg_rate={rate:.1f}/s db={args.sqlite_out}")

if __name__ == "__main__":
    main()