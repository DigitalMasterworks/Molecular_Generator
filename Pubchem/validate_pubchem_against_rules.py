#!/usr/bin/env python3
# validate_pubchem_against_rules.py
#
# Stream every SMILES in CID-SMILES.gz and write a SQLite row for EACH record:
#   cid, smiles, status, reason, ruleset, alpha_pi, ts
# Status values: valid, invalid, parse_fail, kekulize_fail, no_heavy, salt_accept, unsupported, malformed
#
# Zero RAM growth: we parse one line, decide, INSERT, and move on.

import argparse, gzip, sys, time, sqlite3
from typing import Dict, Tuple, List

import molecule_sql_verify as msv  # uses verify_cert(atoms, mult)

def build_atoms_and_mult_from_smiles(smi: str) -> Tuple[List[str], Dict[Tuple[int,int], int]]:
    """SMILES -> (atoms with total H, mult map heavy-heavy). Kekulize to resolve aromaticity."""
    from rdkit import Chem
    from rdkit.Chem import rdchem
    
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
        
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError("parse_fail")

    # Try Kekulize to force explicit single/double; if it fails, keep original
    m2 = Chem.Mol(m)
    try:
        Chem.Kekulize(m2, clearAromaticFlags=True)
    except Exception:
        m2 = m  # we'll still proceed; if verify fails it will be counted

    # heavy atoms map
    heavy_ids = [a.GetIdx() for a in m2.GetAtoms() if a.GetAtomicNum() != 1]
    if not heavy_ids:
        raise ValueError("no_heavy")

    old2new = {old: new for new, old in enumerate(heavy_ids)}

    # total H across heavy atoms (implicit+explicit)
    total_H = 0
    for a in m2.GetAtoms():
        if a.GetAtomicNum() != 1:
            total_H += int(a.GetTotalNumHs())

    # atoms: heavy first, then H * total_H
    atoms = [m2.GetAtomWithIdx(aid).GetSymbol() for aid in heavy_ids]
    atoms.extend(["H"] * total_H)

    # multiplicities
    mult: Dict[Tuple[int,int], int] = {}
    for b in m2.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        if i not in old2new or j not in old2new:
            continue
        ni, nj = old2new[i], old2new[j]
        a, b2 = (ni, nj) if ni < nj else (nj, ni)

        bt = b.GetBondType()
        if   bt == rdchem.BondType.SINGLE:  mval = 1
        elif bt == rdchem.BondType.DOUBLE:  mval = 2
        elif bt == rdchem.BondType.TRIPLE:  mval = 3
        else:                               mval = 1
        # keep max in case of weird duplicates
        prev = mult.get((a,b2), 0)
        if mval > prev: mult[(a,b2)] = mval

    return atoms, mult

def open_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=60.0)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    con.execute("""
        CREATE TABLE IF NOT EXISTS scan_results(
            cid        TEXT NOT NULL,
            smiles     TEXT NOT NULL,
            status     TEXT NOT NULL,   -- valid | invalid | parse_fail | kekulize_fail | no_heavy | salt_accept | unsupported | malformed
            reason     TEXT,            -- verify_cert reason, or short tag matching status
            ruleset    TEXT NOT NULL,
            alpha_pi   REAL NOT NULL,
            ts         INTEGER NOT NULL
        )
    """)
    # Uniqueness per (cid, ruleset, alpha_pi); adjust if you want multi-pass overwrite
    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_scan_key
        ON scan_results(cid, ruleset, alpha_pi)
    """)
    con.execute("CREATE INDEX IF NOT EXISTS ix_status ON scan_results(status)")
    return con

def prepare_atoms_like_db(atoms: List[str], *, q_charge_units: int) -> List[str]:
    """
    Mirror the DB path's per-molecule SALT_Q + H-balance parity fix:
      - set SALT_Q = q_charge_units (positive-charge units on the skeleton)
      - run _hydrogen_balance_fix() once to bring H into identity
      - re-check parity and nudge SALT_Q by ±1 or ±2 if needed (like the DB path)
      - run _hydrogen_balance_fix() again
    Notes:
      - salt_delta_metal=0 and cation_charge=q_charge_units here (no metal cleanup in SMILES path)
      - we DO NOT drop halogens or spectators here; this is the "opposite-way" check on the given SMILES
    """
    from collections import defaultdict

    def _counts(lst):
        c = defaultdict(int)
        for a in lst:
            c[a] += 1
        return c

    # 1) first H-balance pass with SALT_Q = q_charge_units
    saved = msv.SALT_Q
    try:
        msv.SALT_Q = q_charge_units
        atoms1 = msv._hydrogen_balance_fix(
            atoms,
            q_charge_units,
            drop_halos_for_H=False,
            salt_delta_metal=0,
            cation_charge=q_charge_units
        )
        # 2) recompute parity like DB path and nudge SALT_Q if off by 1/2
        counts1 = _counts(atoms1)
        v_map1  = msv.assign_valences_heavy(atoms1)
        v_sum1  = sum(v_map1[i] for i in v_map1.keys())
        nh1     = sum(1 for a in atoms1 if a != "H")
        dbe1    = msv.dbe_from_counts(counts1)   # uses current SALT_Q
        H_need1 = int(v_sum1 - 2*dbe1 - 2*(nh1 - 1))
        H_has1  = counts1.get("H", 0)
        delta   = H_need1 - H_has1

        if delta != 0:
            # odd → ±1; even → ±2·k (cap like DB path)
            if abs(delta) % 2 == 1:
                msv.SALT_Q += (-1 if delta > 0 else +1)
            else:
                steps = min(abs(delta) // 2, 16)
                msv.SALT_Q += (-2 if delta > 0 else +2) * steps

            # 3) second H-balance pass after parity nudge
            atoms1 = msv._hydrogen_balance_fix(
                atoms1,
                msv.SALT_Q,
                drop_halos_for_H=False,
                salt_delta_metal=0,
                cation_charge=q_charge_units
            )
        return atoms1
    finally:
        # do NOT restore SALT_Q here; verify_cert relies on the current value for DBE
        # (verify_cert itself doesn't reset SALT_Q; we're matching DB behavior)
        pass

def recompute_status_for_smiles(smi: str) -> Tuple[str, str]:
    """
    Returns (status, reason) using same logic as streaming:
      parse/kek → atoms,mult → SALT_Q from +charges → prepare_atoms_like_db →
      INORG_ACCEPT guard → verify_cert
    """
    
    # short-circuit: treat common charged motifs as salts in opposite-way path
    if _matches_charged_whitelist(smi):
        return ("salt_accept", "charged_motif")
        
    from rdkit import Chem
    try:
        atoms, mult = build_atoms_and_mult_from_smiles(smi)
    except ValueError as e:
        msg = str(e)
        if msg == "parse_fail": return ("parse_fail", "parse_fail")
        if msg == "no_heavy":   return ("no_heavy",  "no_heavy")
        return ("kekulize_fail", "kekulize_fail")

    # positive charge units from SMILES
    pos_units = 0
    m_tmp = Chem.MolFromSmiles(smi)
    if m_tmp is not None:
        for a in m_tmp.GetAtoms():
            if a.GetAtomicNum() != 1:
                q = a.GetFormalCharge()
                if q > 0: pos_units += q

    atoms = prepare_atoms_like_db(atoms, q_charge_units=pos_units)

    # inorganic/unsupported guard
    others = [a for a in atoms if a not in msv.BASE_VALENCE]
    if others:
        U = set(others)
        INORG = getattr(msv, "INORG_ACCEPT", set())
        if U.issubset(INORG):
            return ("salt_accept", "inorg_whitelist")
        else:
            return ("unsupported", f"unsupported:{','.join(sorted(U))}")

    ok, why = msv.verify_cert(atoms, mult)
    if ok:
        return ("valid", "")
    else:
        return ("invalid", why or "verify_fail")
        
def main():
    ap = argparse.ArgumentParser(description="Validate every SMILES in CID-SMILES.gz under your rules and write a SQLite DB.")
    ap.add_argument("--cid-smiles-gz", help="Path to CID-SMILES.gz")
    ap.add_argument("--sqlite-out", required=True, help="Path to output SQLite (created if missing)")
    ap.add_argument("--ruleset", default="RS3", choices=["RS1","RS2","RS3"])
    ap.add_argument("--alpha-pi", type=float, default=1.15)
    ap.add_argument("--batch", type=int, default=10000, help="DB batch size")
    ap.add_argument("--progress-every-million", type=int, default=1, help="Progress every N million (0=off)")
    ap.add_argument("--truncate", action="store_true", help="Drop previous rows for this (ruleset, alpha_pi) before run")
    ap.add_argument("--recheck-invalids", action="store_true", help="Re-validate ONLY rows currently status='invalid'")
    ap.add_argument("--dry-run", action="store_true", help="Recheck but do not UPDATE rows")
    args = ap.parse_args()

    try:
        from rdkit import Chem  # noqa: F401
    except Exception:
        print("[error] RDKit is required (pip install rdkit-pypi or conda).", file=sys.stderr)
        sys.exit(2)

    # --- RECHECK-ONLY MODE ---
    if args.recheck_invalids:
        con = open_db(args.sqlite_out)
        sel = con.cursor()
        sel.execute("""
            SELECT cid, smiles FROM scan_results
            WHERE status='invalid' AND ruleset=? AND ABS(alpha_pi-?)<1e-12
        """, (args.ruleset, args.alpha_pi))

        upd = con.cursor()
        ts_now = int(time.time())
        total = fixed = same = 0

        rows = sel.fetchmany(args.batch)
        while rows:
            updates = []
            for cid, smi in rows:
                total += 1
                new_status, new_reason = recompute_status_for_smiles(smi)
                if new_status == "invalid": same += 1
                else:                       fixed += 1
                if not args.dry_run:
                    updates.append((new_status, new_reason, ts_now, cid, args.ruleset, args.alpha_pi))
            if updates:
                upd.executemany(
                    "UPDATE scan_results SET status=?, reason=?, ts=? WHERE cid=? AND ruleset=? AND alpha_pi=?",
                    updates
                )
                con.commit()
            rows = sel.fetchmany(args.batch)

        sel.close(); upd.close(); con.close()
        print(f"[recheck] total_invalid_processed={total} fixed_now={fixed} still_invalid={same}")
        return
        
    con = open_db(args.sqlite_out)

    # Optional: clean rows for this config
    if args.truncate:
        con.execute("DELETE FROM scan_results WHERE ruleset=? AND alpha_pi=?", (args.ruleset, args.alpha_pi))
        con.commit()

    total = valid = invalid = parse_fail = kek_fail = no_heavy = 0
    salt_accept = 0
    unsupported = 0

    million_step = args.progress_every_million if args.progress_every_million > 0 else 10**12
    ts_now = int(time.time())

    to_insert = []
    insert_sql = """
        INSERT OR REPLACE INTO scan_results
        (cid, smiles, status, reason, ruleset, alpha_pi, ts)
        VALUES (?,?,?,?,?,?,?)
    """

    def flush():
        if not to_insert:
            return
        con.executemany(insert_sql, to_insert)
        con.commit()
        to_insert.clear()
    
    if not args.cid_smiles_gz:
        print("[error] --cid-smiles-gz is required unless --recheck-invalids is set", file=sys.stderr); sys.exit(2)
        
    with gzip.open(args.cid_smiles_gz, "rt", encoding="utf-8", errors="ignore") as fh:
        for ln, line in enumerate(fh, start=1):
            status = "invalid"
            reason = ""
            try:
                cid, smi = line.rstrip("\n").split("\t", 1)
            except ValueError:
                total += 1
                status = "malformed"
                reason = "no_tab"
                to_insert.append((str(ln), "", status, reason, args.ruleset, args.alpha_pi, ts_now))
                if len(to_insert) >= args.batch: flush()
                continue

            # Parse → (atoms, mult)
            try:
                atoms, mult = build_atoms_and_mult_from_smiles(smi)
            except ValueError as e:
                msg = str(e)
                if msg == "parse_fail":
                    parse_fail += 1; status = "parse_fail"; reason = "parse_fail"
                elif msg == "no_heavy":
                    no_heavy   += 1; status = "no_heavy";   reason = "no_heavy"
                else:
                    kek_fail   += 1; status = "kekulize_fail"; reason = "kekulize_fail"
                total += 1
                to_insert.append((cid, smi, status, reason, args.ruleset, args.alpha_pi, ts_now))
                if len(to_insert) >= args.batch: flush()
                if ln % (million_step * 1_000_000) == 0:
                    print(f"[progress] ~{ln//1_000_000}M | valid={valid} invalid={invalid} "
                          f"| parse_fail={parse_fail} kekulize_fail={kek_fail} no_heavy={no_heavy} "
                          f"| salt_accept={salt_accept} unsupported={unsupported}", file=sys.stderr)
                continue
            
            # Compute positive-charge units from RDKit molecule:
            #   sum of max(0, formal charge) over heavy atoms of the parsed SMILES.
            from rdkit import Chem
            m_tmp = Chem.MolFromSmiles(smi)
            pos_units = 0
            if m_tmp is not None:
                for a in m_tmp.GetAtoms():
                    if a.GetAtomicNum() != 1:
                        q = a.GetFormalCharge()
                        if q > 0:
                            pos_units += q

            # Prepare atoms exactly like DB H-balance path (sets SALT_Q + parity)
            atoms = prepare_atoms_like_db(atoms, q_charge_units=pos_units)
            
            # Inorganic/unsupported guard (mirror DB path)
            others = [a for a in atoms if a not in msv.BASE_VALENCE]
            if others:
                U = set(others)
                INORG = getattr(msv, "INORG_ACCEPT", set())
                if U.issubset(INORG):
                    salt_accept += 1; status = "salt_accept"; reason = "inorg_whitelist"
                else:
                    unsupported += 1; status = "unsupported"; reason = f"unsupported:{','.join(sorted(U))}"
                total += 1
                to_insert.append((cid, smi, status, reason, args.ruleset, args.alpha_pi, ts_now))
                if len(to_insert) >= args.batch: flush()
                if ln % (million_step * 1_000_000) == 0:
                    print(f"[progress] ~{ln//1_000_000}M | valid={valid} invalid={invalid} "
                          f"| parse_fail={parse_fail} kekulize_fail={kek_fail} no_heavy={no_heavy} "
                          f"| salt_accept={salt_accept} unsupported={unsupported}", file=sys.stderr)
                continue

            # Organic core → verify
            ok, why = msv.verify_cert(atoms, mult)
            if ok:
                valid += 1; status = "valid";   reason = ""
            else:
                invalid += 1; status = "invalid"; reason = why or "verify_fail"
            total += 1

            to_insert.append((cid, smi, status, reason, args.ruleset, args.alpha_pi, ts_now))
            if len(to_insert) >= args.batch:
                flush()
            if ln % (million_step * 1_000_000) == 0:
                print(f"[progress] ~{ln//1_000_000}M | valid={valid} invalid={invalid} "
                      f"| parse_fail={parse_fail} kekulize_fail={kek_fail} no_heavy={no_heavy} "
                      f"| salt_accept={salt_accept} unsupported={unsupported}", file=sys.stderr)

    flush()
    print(
        f"[summary] scanned={total} "
        f"valid_under_rules={valid} invalid_under_rules={invalid} "
        f"parse_fail={parse_fail} kekulize_fail={kek_fail} no_heavy={no_heavy} "
        f"salt_accept={salt_accept} unsupported={unsupported} "
        f"db='{args.sqlite_out}'"
    )

if __name__ == "__main__":
    main()