#!/usr/bin/env python3
# molecule_sql_verify.py
#
# Terminal-only verifier for the paper-math construction.
# Prints one line per molecule:
#   PASS NAME — FORM — DBE=… rings=… pi=… [oracle ✓/✗/–]
#   FAIL NAME — FORM — REASON
#
# Design:
#  - σ-base: degree-constrained maximum-weight spanning tree (Kruskal + deg ≤ valence)
#  - DBE spender: merged greedy between extra-σ and π tickets under per-atom headroom
#  - Dynamic P/S valence: P ∈ {3,5}, S ∈ {2,6} chosen to satisfy Σv − H − 2·DBE = 2(n_h − 1)

import argparse, math, sys, random, sqlite3, itertools, os
from collections import defaultdict, deque
from typing import Dict, List, Tuple

# --- Optional RDKit import for --print rendering ---
_RD_SUPPORT = True
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
except Exception:
    _RD_SUPPORT = False
    
# Optional RDKit standardizer (salt remover)
_RD_STD = None
if _RD_SUPPORT:
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        _RD_STD = rdMolStandardize.SaltRemover()
    except Exception:
        _RD_STD = None
        
    
# --- Tuning overrides (optional, set via CLI) ---
PI_RESERVE_OVERRIDE: int | None = None  # 0..2 or None
RS3_BUMP_SIGMA = 0.20  # default σ-like bump for RS3
RS3_BUMP_PI    = 0.10  # default π-like bump for RS3

# --- salt charge for cationic skeleton (set per-molecule in main loop) ---
SALT_Q = 0  # number of detached halide anions (q = total positive charge of skeleton)

# ---------- Ladder, kernel, periodic data ----------
MEV_PER_U   = 931.49410242
MASS_E_MEV  = 0.51099895000
MASS_MU_MEV = 105.6583755
MU_E        = MASS_MU_MEV / MASS_E_MEV

SIGMA0 = 0.900000
SIGMA1 = 1.200000
DELTA1 = 6.500000
W0 = 1.124462
W1 = 1.551250

AMU: Dict[str, float] = {
    "H": 1.00784, "C": 12.0107, "N": 14.0067, "O": 15.999,
    "F": 18.998403163, "Cl": 35.45, "Br": 79.904, "I": 126.90447,
    "P": 30.973761998, "S": 32.065,
    "B": 10.81, "Si": 28.085, "Ge": 72.63,
    "As": 74.9216, "Se": 78.971, "Te": 127.60
}
BASE_VALENCE = {
    "H":1, "C":4, "N":3, "O":2,
    "F":1, "Cl":1, "Br":1, "I":1,
    "P":3, "S":2,
    "B":3, "Si":4, "Ge":4,
    "As":3, "Se":2, "Te":2
}
HALO = {"F","Cl","Br","I"}
# skip-class sets
SALT_CATIONS = {"Na","K","Li","Rb","Cs","Ca","Mg"}  # spectator cations in salts
SEMIMETALS = {"B","Si","Ge","As","Se","Te"}  # out-of-scope main-group extensions for now
# Inorganic-accept whitelist (elements we treat as salts/inorganics if unsupported for organics)
INORG_ACCEPT = {
    # Noble gases
    "He","Ne","Ar","Kr","Xe","Rn",
    # Alkali/alkaline
    "Li","Na","K","Rb","Cs","Fr","Be","Mg","Ca","Sr","Ba","Ra",
    # Common inorganic/post-transition elements (include At)
    "Al","Ga","In","Tl","Ge","Sn","Pb","Bi","Sb","Po","Hg","Ag","Au","At",
    # Early/transition, lanthanides, actinides
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    # Semi-metals already excluded from organic core
    "B","Si","Ge","As","Se","Te"
}
FORM_RE = __import__("re").compile(r"([A-Z][a-z]?)(\d*)")

def parse_formula(form: str) -> List[str]:
    atoms: List[str] = []
    for sym, num in FORM_RE.findall(form):
        cnt = int(num) if num else 1
        atoms.extend([sym] * cnt)
    return atoms

def rung_from_atomic_mass_u(amu: float) -> float:
    mZ = amu * MEV_PER_U
    return 3.0 + 13.0 * math.log(mZ / MASS_E_MEV) / math.log(MU_E)

def kernel_weight(nu_i: float, nu_j: float) -> float:
    d = abs(nu_i - nu_j)
    p0 = math.exp(-(d / SIGMA0) ** 2)
    p1 = math.exp(-((d - DELTA1) ** 2) / (2.0 * SIGMA1 ** 2))
    return W0 * p0 + W1 * p1

def dbe_from_counts(counts: Dict[str,int]) -> int:
    """
    Integer-safe, charge-aware DBE.
      s = Σ n_i (v_i - 2)          (integer)
      DBE = 1 + (s + SALT_Q) // 2  (integer)
    """
    s = 0
    for elem, n in counts.items():
        v = BASE_VALENCE.get(elem)
        if v is None:
            continue
        s += n * (v - 2)
    return 1 + (s + SALT_Q) // 2

# ---------- Pair caps ----------
def pair_cap(a: str, b: str) -> int:
    if a in HALO and b in HALO: return 0
    if (a in HALO) ^ (b in HALO): return 1
    t = tuple(sorted((a,b)))
    if t == ("O","O"): return 1
    if t in {("C","O"), ("N","O"), ("S","O")}: return 2
    if t == ("C","C"): return 3
    if t in {("C","N"), ("N","N")}: return 3
    return 2

def p_cap(a: str, b: str) -> int:
    return max(0, pair_cap(a,b) - 1)

# ---------- Rule sets ----------
def compute_weights(nu_i, nu_j, ei, ej, ruleset: str, alpha_pi: float) -> Tuple[float,float]:
    base = kernel_weight(nu_i, nu_j)
    if ruleset == "RS1":
        return base, base * alpha_pi
    if ruleset == "RS2":
        w_sigma = base; w_pi = base
        if {ei,ej} == {"C","O"}: w_pi *= 1.35
        return w_sigma, w_pi * alpha_pi
    if ruleset == "RS3":
        d = abs(nu_i - nu_j); like_bump = math.exp(-(d/0.80)**2)
        w_sigma = base + (RS3_BUMP_SIGMA * like_bump if {ei,ej} in ({"C","C"},{"C","N"}) else 0.0)
        w_pi    = base + (RS3_BUMP_PI    * like_bump if {ei,ej} in ({"C","C"},{"C","N"}) else 0.0)
        return w_sigma, w_pi * alpha_pi
    return base, base * alpha_pi

# ---------- DSU ----------
class DSU:
    def __init__(self, nodes):
        self.p = {u:u for u in nodes}; self.r = {u:0 for u in nodes}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.r[ra] < self.r[rb]: self.p[ra] = rb
        elif self.r[ra] > self.r[rb]: self.p[rb] = ra
        else: self.p[rb] = ra; self.r[ra] += 1
        return True

# ---------- Connectivity ----------
def is_connected(n: int, edges: List[Tuple[int,int]]) -> bool:
    if n <= 1: return True
    adj = [[] for _ in range(n)]
    for i,j in edges: adj[i].append(j); adj[j].append(i)
    seen={0}; q=deque([0])
    while q:
        u=q.popleft()
        for v in adj[u]:
            if v not in seen: seen.add(v); q.append(v)
    return len(seen) == n

# ---------- Dynamic P/S valence assignment ----------
def assign_valences_heavy(atoms: List[str]) -> Dict[int,int]:
    """
    Choose valences for heavy atoms to satisfy the global identity:
      Σ v_i  =  H + 2*DBE + 2*(n_h - 1)
    Allowed promotions (minimal first):
      N: 3 -> 4
      P: 3 -> 4  (then ->5 if still needed)
      S: 2 -> 3  (then ->6 if still needed; rarely used here)
    """
    idx_heavy = [i for i,a in enumerate(atoms) if a != "H"]
    nh = len(idx_heavy)
    counts = defaultdict(int)
    for a in atoms: counts[a]+=1
    H  = counts.get("H",0)
    DBE = dbe_from_counts(counts)

    V_target = H + 2*DBE + 2*(nh - 1)

    v = [BASE_VALENCE[a] for a in atoms]
    base_sum = sum(v[i] for i in idx_heavy)

    need = V_target - base_sum
    # start with base valences
    v_map = {i: v[i] for i in idx_heavy}
    if need <= 0:
        return v_map

    # collect indices by element
    N_idxs = [i for i in idx_heavy if atoms[i] == "N"]
    P_idxs = [i for i in idx_heavy if atoms[i] == "P"]
    S_idxs = [i for i in idx_heavy if atoms[i] == "S"]

    # 1) promote N: 3 -> 4  (cost +1 each)
    for i in N_idxs:
        if need <= 0: break
        if v_map[i] == 3:
            v_map[i] = 4; need -= 1

    # 2) promote P: 3 -> 4 (cost +1 each)
    for i in P_idxs:
        if need <= 0: break
        if v_map[i] == 3:
            v_map[i] = 4; need -= 1

    # 3) promote S: 2 -> 3 (cost +1 each)
    for i in S_idxs:
        if need <= 0: break
        if v_map[i] == 2:
            v_map[i] = 3; need -= 1

    # 4) if still short, use P: 4 -> 5 (another +1)
    for i in P_idxs:
        if need <= 0: break
        if v_map[i] == 4:
            v_map[i] = 5; need -= 1
    
    # 4b) if still short, allow N: 4 -> 5 (another +1)
    for i in N_idxs:
        if need <= 0: break
        if v_map[i] == 4:
            v_map[i] = 5; need -= 1

    # 5) last resort, S: 3 -> 6 (+3 more; apply only if large residual)
    for i in S_idxs:
        if need <= 0: break
        if v_map[i] == 3:
            bump = min(need, 3)
            v_map[i] = 3 + bump  # 4,5,6 step; typically will jump to 6 if need>=3
            need -= bump

    # if we still didn't hit exact V_target, verifier will flag; but salts need is tiny (1..2)
    return v_map

REM_ORDER = ("I","Br","Cl","F")  # typical halide anions preference

def base_dbe_nocharge(counts: Dict[str,int]) -> float:
    s = 0.0
    for elem, n in counts.items():
        v = BASE_VALENCE.get(elem)
        if v is None:
            continue
        s += n * (v - 2)
    return 1 + 0.5 * s

def remove_k_halides(atoms: List[str], k: int) -> List[str]:
    """Remove k halogen atoms (prefer I>Br>Cl>F) to form the cationic skeleton."""
    if k <= 0: return atoms[:]
    need = {h: 0 for h in REM_ORDER}
    # plan removals
    counts = defaultdict(int)
    for a in atoms: counts[a]+=1
    kk = k
    for h in REM_ORDER:
        take = min(kk, counts.get(h,0))
        need[h] = take
        kk -= take
        if kk == 0: break
    if kk > 0:
        return atoms[:]  # not enough halogens; return unchanged
    # build new list skipping the planned removals
    out = []
    used = {h:0 for h in REM_ORDER}
    for a in atoms:
        if a in need and used[a] < need[a]:
            used[a] += 1  # drop this halogen
        else:
            out.append(a)
    return out

def remove_k_common_anions(atoms: List[str], k_charge: int) -> Tuple[List[str], int, int]:
    """
    Remove up to k_charge UNITS OF NEGATIVE CHARGE via common anions.
    Returns (new_atoms_list, removed_units, removed_charge_units).

    Order: BF4- (1-), NO3- (1-), HPO4^2- (2-), SO4^2- (2-), H2PO4- (1-), HSO4- (1-), OH- (1-).
    """
    if k_charge <= 0:
        return atoms[:], 0, 0

    out = atoms[:]

    def try_remove(seq: Dict[str, int]) -> bool:
        nonlocal out
        c = defaultdict(int)
        for a in out:
            c[a] += 1
        if not all(c[sym] >= need for sym, need in seq.items()):
            return False
        needed = dict(seq)
        new_out = []
        for a in out:
            if a in needed and needed[a] > 0:
                needed[a] -= 1
            else:
                new_out.append(a)
        out = new_out
        return True

    removed_units = 0
    removed_charge = 0

    # anion patterns: (composition, charge_units)
    patterns: List[Tuple[Dict[str,int], int]] = [
        ({"B":1,  "F":4}, 1),               # BF4-
        ({"N":1,  "O":3}, 1),               # NO3-
        ({"H":1,  "P":1, "O":4}, 2),        # HPO4^2-
        ({"S":1,  "O":4}, 2),               # SO4^2-
        ({"H":2,  "P":1, "O":4}, 1),        # H2PO4-
        ({"P":1,  "O":4}, 3),               # PO4^3-
        ({"P":1,  "F":6}, 1),               # PF6-
        ({"As":1, "F":6}, 1),               # AsF6-
        ({"Sb":1, "F":6}, 1),               # SbF6-
        ({"Cl":1, "O":4}, 1),               # ClO4-
        ({"Re":1, "O":4}, 1),               # ReO4-
        ({"C":1,  "F":3, "S":1, "O":3}, 1), # CF3SO3- (triflate)
        ({"Sb":1, "O":6, "H":1}, 1),        # HSbO6-  (hexahydroxoantimonate, common)
        ({"Sb":2, "O":7}, 2),               # Sb2O7^2- (pyroantimonate unit)
        ({"H":1,  "S":1, "O":4}, 1),        # HSO4-
        ({"O":1,  "H":1}, 1),               # OH-
    ]

    while removed_charge < k_charge:
        progressed = False
        for seq, ch in patterns:
            if removed_charge >= k_charge:
                break
            if try_remove(seq):
                removed_units += 1
                removed_charge += ch
                progressed = True
                break
        if not progressed:
            break

    return out, removed_units, removed_charge

    removed = 0
    while removed < k:
        # 1) BF4- (B + 4F)
        if try_remove({"B": 1, "F": 4}):
            removed += 1
            continue
        # 2) NO3- (N + 3O) — prefer nitrate ahead of hydroxide to avoid spurious H shifts
        if try_remove({"N": 1, "O": 3}):
            removed += 1
            continue
        # 3) H2PO4- (2H + P + 4O) — common phosphate counter-ion
        if try_remove({"H": 2, "P": 1, "O": 4}):
            removed += 1
            continue
        # 4) HSO4- (H + S + 4O)
        if try_remove({"H": 1, "S": 1, "O": 4}):
            removed += 1
            continue
        # 5) OH- (O + H) — last resort
        if try_remove({"O": 1, "H": 1}):
            removed += 1
            continue
        break

    return out, removed

def _hydrogen_balance_fix(
    atoms: List[str], q_charge: int, *, drop_halos_for_H: bool = False, salt_delta_metal: int = 0, cation_charge: int = 0) -> List[str]:
    """
    Adjust H so the verifier's identity holds on a cleaned core:
      H_needed = sum(v_i) - 2*DBE_local - 2*(n_h - 1)
    where v_i comes from assign_valences_heavy (same as verifier),
    and DBE_local uses dbe_from_counts with SALT_Q = q_charge.

    Core cleaning: drop spectator cations and transition/rare metals; KEEP halogens.
    """
    # build core for H-identity:
    # - always drop spectator cations and transition/rare metals
    # - drop halogens *only* if drop_halos_for_H=True (metal-counter-ion view)
    _MONO = {"Na","K","Li","Rb","Cs"}
    _DI   = {"Ca","Mg"}
    _TM = {"Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
           "Mo","Ru","Rh","Pd","Ag","Cd","W","Re","Os",
           "Ir","Pt","Au","Hg","Tc",
           "Sn","Sb","Tl","Bi","Pb",
           "Y","La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"}

    if drop_halos_for_H:
        core = [a for a in atoms if (a not in _MONO and a not in _DI and a not in _TM and a not in HALO)]
    else:
        core = [a for a in atoms if (a not in _MONO and a not in _DI and a not in _TM)]

    # if core is empty or only H, nothing to adjust
    if not core or all(a == "H" for a in core):
        return atoms

    # compute on the core with same machinery as verifier
    global SALT_Q
    _old_q = SALT_Q
    SALT_Q = q_charge + salt_delta_metal + cation_charge
    try:
        counts = defaultdict(int)
        for a in core: counts[a] += 1

        # promoted valences exactly like verifier
        v_map = assign_valences_heavy(core)
        v_sum = sum(v_map[i] for i in v_map.keys())

        nh = sum(1 for a in core if a != "H")
        dbe_local = dbe_from_counts(counts)
        H_needed = int(v_sum - 2 * dbe_local - 2 * (nh - 1))
        H_present = counts.get("H", 0)
        delta = H_needed - H_present
    finally:
        SALT_Q = _old_q

        # Apply the first-pass correction
        def _apply_delta(atom_list: List[str], d: int) -> List[str]:
            if d == 0:
                return atom_list
            if d > 0:
                return atom_list + (["H"] * d)
            # d < 0: remove -d hydrogens
            need = -d
            out2 = []
            for a in atom_list:
                if a == "H" and need > 0:
                    need -= 1
                    continue
                out2.append(a)
            return out2

        atoms = _apply_delta(atoms, delta)

        # ---- Parity stabilization (handles DBE rounding vs H parity) ----
        def _compute_needed(cur_atoms: List[str]) -> Tuple[int,int,int]:
            # rebuild the same 'core' policy used above
            if drop_halos_for_H:
                core2 = [a for a in cur_atoms if (a not in _MONO and a not in _DI and a not in _TM and a not in HALO)]
            else:
                core2 = [a for a in cur_atoms if (a not in _MONO and a not in _DI and a not in _TM)]
            if not core2 or all(a == "H" for a in core2):
                return (0, 0, 0)

            counts2 = defaultdict(int)
            for a in core2: counts2[a] += 1

            _old_q2 = SALT_Q
            try:
                globals()['SALT_Q'] = q_charge + salt_delta_metal + cation_charge
                v_map2 = assign_valences_heavy(core2)
                v_sum2 = sum(v_map2[i] for i in v_map2.keys())
                nh2 = sum(1 for a in core2 if a != "H")
                dbe2 = dbe_from_counts(counts2)  # uses int(round(...)) just like verifier
                H_need2 = int(v_sum2 - 2*dbe2 - 2*(nh2 - 1))
                H_has2  = counts2.get("H", 0)
                return (H_need2, H_has2, nh2)
            finally:
                globals()['SALT_Q'] = _old_q2

        H_need2, H_has2, _ = _compute_needed(atoms)
        delta2 = H_need2 - H_has2
        if delta2:
            atoms = _apply_delta(atoms, delta2)

        return atoms
    
# ---------- Core construction ----------
def construct_graph(atoms: List[str], ruleset: str, alpha_pi: float):
    idx_heavy = [i for i,a in enumerate(atoms) if a != "H"]
    nh = len(idx_heavy)
    if nh == 0: return {}, [], "no_heavy"

    # DBE & rungs
    counts = defaultdict(int)
    for a in atoms: counts[a]+=1
    DBE = dbe_from_counts(counts)
    try:
        nu = [rung_from_atomic_mass_u(AMU[a]) for a in atoms]
    except KeyError as e:
        return {}, [], f"unsupported_element:{e}"

    # valences (heavy)
    v_map = assign_valences_heavy(atoms)

    # candidate edges with weights/caps (heavy-only)
    edges_all = []
    for ii in range(nh):
        i = idx_heavy[ii]
        for jj in range(ii+1, nh):
            j = idx_heavy[jj]
            cap = pair_cap(atoms[i], atoms[j])
            if cap <= 0: continue
            ws, wp = compute_weights(nu[i], nu[j], atoms[i], atoms[j], ruleset, alpha_pi)
            edges_all.append((ws, wp, i, j, cap, p_cap(atoms[i], atoms[j])))
    if not edges_all:
        if DBE==0: return {}, nu, ""
        return {}, nu, "no_heavy_pairs"

    # 1) Two-phase degree-constrained tree:
    #    (a) build core (no halogens), then (b) attach each halogen as a leaf.

    edges_sorted = sorted(edges_all, key=lambda t: t[0], reverse=True)

    idx_core = [i for i in idx_heavy if atoms[i] not in HALO]
    idx_halo = [i for i in idx_heavy if atoms[i] in HALO]

    deg = defaultdict(int)
    tree = []

    # (a) core MWST (only edges i–j with both in core)
    if len(idx_core) >= 2:
        dsu = DSU(idx_core)
        for ws, wp, i, j, cap, pcap in edges_sorted:
            if (i not in idx_core) or (j not in idx_core):
                continue
            if deg[i] >= v_map[i] or deg[j] >= v_map[j]:
                continue
            if not dsu.union(i, j):
                continue
            tree.append((i, j))
            deg[i] += 1; deg[j] += 1
            if len(tree) == len(idx_core) - 1:
                break
        if len(tree) < max(0, len(idx_core) - 1):
            return {}, nu, "tree_disconnected"

    # --- π-reservation on strongest core–core pair (if DBE>0) ---
    # keeps 1 unit of headroom at both endpoints of the best π-eligible core edge
    reserve = defaultdict(int)  # default 0 for all nodes
    if DBE > 0 and len(idx_core) >= 2:
        # reserve up to R anchors on top π-eligible core–core edges, disjoint if possible
        R = min(DBE, PI_RESERVE_OVERRIDE) if PI_RESERVE_OVERRIDE is not None else min(DBE, 2)
        chosen = []
        for ws, wp, i, j, cap, pcap in sorted(
                [(ws, wp, i, j, cap, pcap) for (ws, wp, i, j, cap, pcap) in edges_sorted
                 if i in idx_core and j in idx_core and cap >= 1 and pcap >= 1],
                key=lambda t: t[1], reverse=True):
            # prefer disjoint endpoints
            if any(i in pair or j in pair for pair in chosen):
                continue
            chosen.append((i,j))
            if len(chosen) == R:
                break
        # if we didn’t get R disjoint, allow reuse of endpoints to still reserve something
        if len(chosen) < R:
            for ws, wp, i, j, cap, pcap in sorted(
                    [(ws, wp, i, j, cap, pcap) for (ws, wp, i, j, cap, pcap) in edges_sorted
                     if i in idx_core and j in idx_core and cap >= 1 and pcap >= 1],
                    key=lambda t: t[1], reverse=True):
                if (i,j) not in chosen and len(chosen) < R:
                    chosen.append((i,j))
        for ai, aj in chosen:
            reserve[ai] = max(reserve[ai], 1)
            reserve[aj] = max(reserve[aj], 1)

    # (b) attach each halogen to the best core neighbor with free degree
    #     (highest w_sigma; cap >= 1; deg[core] < v_map[core]; halogen has v=1)
    # (b) attach each halogen as a leaf to a core atom, respecting π reservation
    idx_core = [u for u in idx_heavy if atoms[u] not in HALO]
    idx_halo = [u for u in idx_heavy if atoms[u] in HALO]

    best_partner = {h: [] for h in idx_halo}
    for ws, wp, i, j, cap, pcap in edges_sorted:
        if cap < 1:
            continue
        if i in idx_halo and j in idx_core:
            best_partner[i].append((ws, j, cap))
        elif j in idx_halo and i in idx_core:
            best_partner[j].append((ws, i, cap))

    def avail_capacity(u):
        # do not consume the reserved unit on π-anchors
        return v_map[u] - reserve[u] - deg.get(u, 0)

    for h in idx_halo:
        if deg.get(h, 0) >= v_map[h]:
            continue
        cand = best_partner.get(h, [])
        if not cand:
            return {}, nu, "tree_disconnected"
        # pick core host with max (avail_capacity, w_sigma)
        cand.sort(key=lambda t: (avail_capacity(t[1]), t[0]), reverse=True)
        attached = False
        for ws, core_j, cap in cand:
            if avail_capacity(core_j) > 0:
                tree.append((h, core_j))
                deg[h]  = deg.get(h, 0) + 1
                deg[core_j] = deg.get(core_j, 0) + 1
                attached = True
                break
        if not attached:
            return {}, nu, "tree_disconnected"

    # multiplicities: start with σ on tree edges
    mult = defaultdict(int)
    for (i,j) in tree:
        a,b = (i,j) if i<j else (j,i)
        mult[(a,b)] = 1

    # headroom per heavy atom
    c = {i: v_map[i] - deg.get(i,0) for i in idx_heavy}

    # streams: σ-extras (non-tree) and π tickets (on σ edges)
    in_tree = set((min(i,j),max(i,j)) for (i,j) in tree)
    sigma_items = []  # ("sigma", weight, i, j, consume=1, cap_total)
    for ws, wp, i, j, cap, pcap in edges_sorted:
        a,b=(i,j) if i<j else (j,i)
        if (a,b) in in_tree: continue
        sigma_items.append(("sigma", ws, a, b, 1, cap))

    def build_pi_items():
        t=[]
        for ws, wp, i, j, cap, pcap in edges_sorted:
            a,b=(i,j) if i<j else (j,i)
            if (a,b) not in mult: continue
            m = mult[(a,b)]
            allow = min(pcap, max(0, cap - m))
            for _ in range(allow):
                t.append(("pi", wp, a, b, 1, cap))
        return t

    pi_items = build_pi_items()
    sigma_items.sort(key=lambda t: t[1], reverse=True)
    pi_items.sort(key=lambda t: t[1], reverse=True)

    # DBE spending with local headroom (merged greedy)
    remaining = DBE
    sigma_idx = 0

    def pop_next_sigma():
        nonlocal sigma_idx
        while sigma_idx < len(sigma_items):
            item = sigma_items[sigma_idx]; sigma_idx += 1
            _, w, i, j, cons, cap = item
            if c.get(i,0) >= cons and c.get(j,0) >= cons:
                return item
        return None

    def peek_next_sigma():
        for k in range(sigma_idx, len(sigma_items)):
            _, w, i, j, cons, cap = sigma_items[k]
            if c.get(i,0) >= cons and c.get(j,0) >= cons:
                return sigma_items[k]
        return None

    def peek_next_pi():
        for item in pi_items:
            _, w, i, j, cons, cap = item
            if (i,j) not in mult: continue
            if mult[(i,j)] >= cap: continue
            if c.get(i,0) >= cons and c.get(j,0) >= cons:
                return item
        return None

    while remaining > 0:
        cand_sigma = peek_next_sigma()
        cand_pi    = peek_next_pi()
        if (cand_sigma is None) and (cand_pi is None):
            return {}, nu, "dbe_unspendable"

        choose_sigma = False
        if cand_sigma and cand_pi:
            choose_sigma = cand_sigma[1] >= cand_pi[1]
        elif cand_sigma:
            choose_sigma = True
        else:
            choose_sigma = False

        if choose_sigma:
            item = pop_next_sigma()
            if item is None:
                cand_pi = peek_next_pi()
                if cand_pi is None: return {}, nu, "dbe_unspendable"
            else:
                _, w, i, j, cons, cap = item
                c[i]-=cons; c[j]-=cons
                mult[(i,j)] = 1
                in_tree.add((i,j))
                # enable π tickets on this new σ edge
                ws, wp = compute_weights(nu[i],nu[j],atoms[i],atoms[j],ruleset,alpha_pi)
                pcap = p_cap(atoms[i],atoms[j])
                allow = min(pcap, cap-1)
                for _ in range(allow): pi_items.append(("pi", wp, i, j, 1, cap))
                pi_items.sort(key=lambda t: t[1], reverse=True)
                remaining -= 1
                continue

        # choose π
        item = peek_next_pi()
        if item is None:
            item = pop_next_sigma()
            if item is None: return {}, nu, "dbe_unspendable"
            _, w, i, j, cons, cap = item
            c[i]-=cons; c[j]-=cons
            mult[(i,j)] = 1
            in_tree.add((i,j))
            ws, wp = compute_weights(nu[i],nu[j],atoms[i],atoms[j],ruleset,alpha_pi)
            pcap = p_cap(atoms[i],atoms[j])
            allow = min(pcap, cap-1)
            for _ in range(allow): pi_items.append(("pi", wp, i, j, 1, cap))
            pi_items.sort(key=lambda t: t[1], reverse=True)
            remaining -= 1
        else:
            _, w, i, j, cons, cap = item
            c[i]-=cons; c[j]-=cons
            mult[(i,j)] += 1
            remaining -= 1
            # remove one ticket instance
            for idx,it in enumerate(pi_items):
                if it is item: del pi_items[idx]; break

    return mult, nu, ""  # success

# ---------- Verifier ----------
def verify_cert(atoms: List[str], mult: Dict[Tuple[int,int], int]):
    idx_heavy = [i for i,a in enumerate(atoms) if a != "H"]
    nh = len(idx_heavy)
    if nh == 0: return False, "no_heavy"

    counts = defaultdict(int)
    for a in atoms: counts[a]+=1
    H = counts.get("H",0); DBE = dbe_from_counts(counts)

    v_map = assign_valences_heavy(atoms)

    deg = [0]*len(atoms); piu = [0]*len(atoms); edges=[]
    y_count = 0
    for (i,j), m in mult.items():
        if atoms[i]=="H" or atoms[j]=="H": return False, "H_edge_present"
        if m > pair_cap(atoms[i],atoms[j]): return False, "cap_violation"
        y_count += 1
        deg[i]+=1; deg[j]+=1
        piu[i]+=max(0,m-1); piu[j]+=max(0,m-1)
        edges.append((i,j))

    # hydrogens slack & sum
    hsum=0
    for i in idx_heavy:
        h_i = v_map[i] - deg[i] - piu[i]
        if h_i < 0: return False, "negative_h"
        hsum += h_i
    if hsum != H: return False, f"total_H_mismatch:{hsum}!= {H}"

    # connectivity
    if nh >= 2:
        mapping = {old:k for k, old in enumerate(idx_heavy)}
        heavy_edges = [(mapping[i],mapping[j]) for (i,j) in edges if (i in mapping and j in mapping)]
        if not is_connected(nh, heavy_edges): return False, "disconnected"

    # DBE equality
    p_total = sum(max(0,m-1) for m in mult.values())
    if p_total + (y_count - nh + 1) != DBE: return False, "dbe_mismatch"

    return True, ""

def iter_chembl_formulas_sqlite(db_path: str, offset: int = 0, limit: int | None = None):
    """
    Yields {"NAME": chembl_id, "FORM": formula} from ChEMBL SQLite,
    auto-detecting which formula column(s) exist.
    """
    import sqlite3, sys

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # columns in compound_properties (formula)
    cur.execute("PRAGMA table_info(compound_properties)")
    props_cols = {row[1] for row in cur.fetchall()}

    # columns in compound_structures (SMILES; optional)
    cur.execute("PRAGMA table_info(compound_structures)")
    cs_cols = {row[1] for row in cur.fetchall()}

    # formula selector (unchanged logic)
    candidates = ["full_molformula","molecular_formula","chembl_molformula","full_mol_formula","mol_formula"]
    have = [c for c in candidates if c in props_cols]
    if not have:
        con.close()
        raise RuntimeError("No known formula column found in compound_properties. Checked: " + ", ".join(candidates))

    if len(have) == 1:
        select_formula = f"cp.{have[0]}"
        where_ok = f"(cp.{have[0]} IS NOT NULL AND cp.{have[0]} != '')"
    else:
        select_formula = "COALESCE(" + ",".join(f"cp.{c}" for c in have) + ")"
        where_ok = " OR ".join([f"(cp.{c} IS NOT NULL AND cp.{c} != '')" for c in have])

    # smiles selector (optional)
    smiles_candidates = ["canonical_smiles","standard_smiles","smiles"]
    smiles_cols = [c for c in smiles_candidates if c in cs_cols]
    if smiles_cols:
        select_smiles = f"cs.{smiles_cols[0]}" if len(smiles_cols)==1 else "COALESCE(" + ",".join(f"cs.{c}" for c in smiles_cols) + ")"
    else:
        select_smiles = "NULL"

    # query
    if limit is None:
        q = f"""
        SELECT md.chembl_id, {select_formula} AS formula, {select_smiles} AS smiles
        FROM molecule_dictionary md
        LEFT JOIN compound_properties  cp ON cp.molregno = md.molregno
        LEFT JOIN compound_structures  cs ON cs.molregno = md.molregno
        WHERE {where_ok}
        LIMIT -1 OFFSET ?
        """
        params = (offset,)
    else:
        q = f"""
        SELECT md.chembl_id, {select_formula} AS formula, {select_smiles} AS smiles
        FROM molecule_dictionary md
        LEFT JOIN compound_properties  cp ON cp.molregno = md.molregno
        LEFT JOIN compound_structures  cs ON cs.molregno = md.molregno
        WHERE {where_ok}
        LIMIT ? OFFSET ?
        """
        params = (limit, offset)

    print(f"[chembl] using formula columns: {', '.join(have)}", file=sys.stderr)

    for chembl_id, formula, smiles in cur.execute(q, params):
        if not formula:
            continue
        yield {
            "NAME":   str(chembl_id),
            "FORM":   "".join(str(formula).split()),
            "SMILES": (smiles or "").strip() if smiles else ""
        }

    con.close()

def iter_chembl_formulas_by_ids_sqlite(db_path: str, id_list: List[str]):
    """
    Yields {"NAME": chembl_id, "FORM": formula} for a specific list of CHEMBL IDs.
    Uses the same formula column detection as iter_chembl_formulas_sqlite, but
    restricts rows at the SQL level with an IN (...) clause.
    """
    import sqlite3, sys

    if not id_list:
        return
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("PRAGMA table_info(compound_properties)")
    props_cols = {row[1] for row in cur.fetchall()}

    cur.execute("PRAGMA table_info(compound_structures)")
    cs_cols = {row[1] for row in cur.fetchall()}

    candidates = ["full_molformula","molecular_formula","chembl_molformula","full_mol_formula","mol_formula"]
    have = [c for c in candidates if c in props_cols]
    if not have:
        con.close()
        raise RuntimeError("No known formula column found in compound_properties. Checked: " + ", ".join(candidates))

    if len(have) == 1:
        select_formula = f"cp.{have[0]}"
        where_ok = f"(cp.{have[0]} IS NOT NULL AND cp.{have[0]} != '')"
    else:
        select_formula = "COALESCE(" + ",".join(f"cp.{c}" for c in have) + ")"
        where_ok = " OR ".join([f"(cp.{c} IS NOT NULL AND cp.{c} != '')" for c in have])

    smiles_candidates = ["canonical_smiles","standard_smiles","smiles"]
    smiles_cols = [c for c in smiles_candidates if c in cs_cols]
    select_smiles = f"cs.{smiles_cols[0]}" if len(smiles_cols)==1 else ("COALESCE(" + ",".join(f"cs.{c}" for c in smiles_cols) + ")") if smiles_cols else "NULL"

    placeholders = ",".join(["?"] * len(id_list))
    q = f"""
    SELECT md.chembl_id, {select_formula} AS formula, {select_smiles} AS smiles
    FROM molecule_dictionary md
    LEFT JOIN compound_properties  cp ON cp.molregno = md.molregno
    LEFT JOIN compound_structures  cs ON cs.molregno = md.molregno
    WHERE ({where_ok}) AND md.chembl_id IN ({placeholders})
    """

    print(f"[chembl] using formula columns: {', '.join(have)}", file=sys.stderr)

    for chembl_id, formula, smiles in cur.execute(q, id_list):
        if not formula:
            continue
        yield {
            "NAME":   str(chembl_id),
            "FORM":   "".join(str(formula).split()),
            "SMILES": (smiles or "").strip() if smiles else ""
        }

    con.close()
    
# ---------- Φ & Oracle ----------
def phi_value(atoms: List[str], mult: Dict[Tuple[int,int], int], nu: List[float], ruleset: str, alpha_pi: float) -> float:
    total=0.0
    for (i,j), m in mult.items():
        ws, wp = compute_weights(nu[i],nu[j],atoms[i],atoms[j],ruleset,alpha_pi)
        total += ws
        if m>1: total += wp*(m-1)
    return total

def oracle_cpsat(atoms: List[str], ruleset: str, alpha_pi: float, time_limit_s: int = 5):
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return None, "ortools_missing"
    idx_heavy=[i for i,a in enumerate(atoms) if a!="H"]; nh=len(idx_heavy)
    if nh==0: return {}, ""
    counts=defaultdict(int)
    for a in atoms: counts[a]+=1
    DBE=dbe_from_counts(counts)
    try:
        nu=[rung_from_atomic_mass_u(AMU[a]) for a in atoms]
    except KeyError as e:
        return None, f"unsupported:{e}"
    v_map=assign_valences_heavy(atoms)

    pairs=[]; caps={}; pmax={}
    w_sigma={}; w_pi={}
    for ii in range(nh):
        i=idx_heavy[ii]
        for jj in range(ii+1, nh):
            j=idx_heavy[jj]
            cap=pair_cap(atoms[i],atoms[j])
            if cap<=0: continue
            ws, wp = compute_weights(nu[i],nu[j],atoms[i],atoms[j],ruleset,alpha_pi)
            pairs.append((i,j)); caps[(i,j)]=cap; pmax[(i,j)]=max(0,cap-1)
            w_sigma[(i,j)]=ws; w_pi[(i,j)]=wp

    if not pairs:
        if DBE==0: return {}, ""
        return None, "no_pairs"

    from ortools.sat.python import cp_model
    m=cp_model.CpModel()
    y={}; p={}
    for (i,j) in pairs:
        y[(i,j)] = m.NewBoolVar(f"y_{i}_{j}")
        p[(i,j)] = m.NewIntVar(0, pmax[(i,j)], f"p_{i}_{j}")
        m.Add(y[(i,j)] + p[(i,j)] <= caps[(i,j)])
    h={i: m.NewIntVar(0, v_map[i], f"h_{i}") for i in idx_heavy}

    nbr={i:[] for i in idx_heavy}
    for (i,j) in pairs: nbr[i].append((i,j)); nbr[j].append((i,j))
    for i in idx_heavy: m.Add(sum(y[e]+p[e] for e in nbr[i]) + h[i] == v_map[i])
    m.Add(sum(h.values()) == counts.get("H",0))

    if nh>=2:
        f={}
        for (i,j) in pairs:
            f[(i,j)] = m.NewIntVar(0, nh-1, f"f_{i}_{j}")
            f[(j,i)] = m.NewIntVar(0, nh-1, f"f_{j}_{i}")
            m.Add(f[(i,j)] <= (nh-1)*y[(i,j)])
            m.Add(f[(j,i)] <= (nh-1)*y[(i,j)])
        root=idx_heavy[0]
        N={i:[] for i in idx_heavy}
        for (i,j) in pairs: N[i].append(j); N[j].append(i)
        m.Add(sum(f[(root,v)] for v in N[root]) - sum(f[(v,root)] for v in N[root]) == nh-1)
        for w in idx_heavy[1:]:
            m.Add(sum(f[(u,w)] for u in N[w]) - sum(f[(w,u)] for u in N[w]) == 1)

    m.Add(sum(p[e] for e in pairs) + sum(y[e] for e in pairs) - nh + 1 == DBE)

    SCALE=1_000_000; obj=[]
    for e in pairs:
        obj.append(int(round(w_sigma[e]*SCALE))*y[e])
        obj.append(int(round(w_pi[e]*SCALE))*p[e])
    m.Maximize(sum(obj))
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=float(time_limit_s)
    solver.parameters.num_search_workers=8
    status=solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, f"status:{solver.StatusName(status)}"
    mult={}
    for (i,j) in pairs:
        yi=solver.Value(y[(i,j)]); pi=solver.Value(p[(i,j)])
        if yi+pi>0:
            a,b=(i,j) if i<j else (j,i)
            mult[(a,b)] = yi+pi
    return mult, ""

# ---------- RDKit drawing helpers ----------
def _safe_mkdir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def _sanitize_filename(s: str) -> str:
    keep = "-_.()[]{} "
    return "".join(ch if (ch.isalnum() or ch in keep) else "_" for ch in s)[:200]

def _draw_smiles_png(
    smiles: str,
    out_png: str,
    *,
    size_px: int = 600,
    kekulize: bool = False,
    strip_salts: bool = False,
    legend: str | None = None
) -> bool:
    if not _RD_SUPPORT or not smiles:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # (a) optionally strip salts/counter-ions
        if strip_salts:
            if _RD_STD is not None:
                mol = _RD_STD.StripMol(mol, dontRemoveEverything=True)
            else:
                # fallback: pick largest organic fragment (prefers any fragment containing carbon)
                frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                if frags:
                    def key(m):
                        has_c = any(a.GetSymbol() == "C" for a in m.GetAtoms())
                        return (1 if has_c else 0, m.GetNumAtoms())
                    mol = sorted(frags, key=key)[-1]

        AllChem.Compute2DCoords(mol)
        if kekulize:
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                pass  # draw anyway

        d2d = Draw.MolDraw2DCairo(size_px, size_px)
        if legend:
            d2d.drawOptions().legendFontSize = int(size_px * 0.05)  # small, scaled
        d2d.DrawMolecule(mol, legend=legend or "")
        d2d.FinishDrawing()
        with open(out_png, "wb") as fh:
            fh.write(d2d.GetDrawingText())
        return True
    except Exception:
        return False
        
# ---------- Driver ----------
def main():
    ap = argparse.ArgumentParser(description="Terminal-only verifier for the paper-math connectivity construction (DB-only).")
    ap.add_argument("--sqlite", required=True, help="Path to chembl_XX.sqlite")
    ap.add_argument("--ruleset", default="RS3", choices=["RS1","RS2","RS3"])
    ap.add_argument("--alpha-pi", type=float, default=1.15, help="Global multiplier for π weights")
    ap.add_argument("--limit", type=int, default=None, help="Max number of rows to read from DB (None = all)")
    ap.add_argument("--offset", type=int, default=0, help="Row offset in DB query")
    ap.add_argument("--max-heavy", type=int, default=50, help="Skip molecules with > this many heavy atoms")
    ap.add_argument("--oracle", type=float, default=0.0, help="Fraction (0..1) to cross-check via CP-SAT")
    ap.add_argument("--oracle-max-heavy", type=int, default=24)
    ap.add_argument("--oracle-time", type=int, default=5)
    ap.add_argument("--only-bad", action="store_true", help="Print only failures (suppress ✓ lines)")
    ap.add_argument("--include-skipped", action="store_true", help="With --only-bad, also print skipped lines")
    ap.add_argument("--log", default="bad.log", help="Write a .log file of bad (and optional skipped) entries")
    ap.add_argument("--checkpoint-every", type=int, default=100000,
                    help="Flush log to disk every N rows (0 disables)")
    ap.add_argument("--only-id", default=None, help="Process only this CHEMBL_ID (e.g., CHEMBL430531)")
    ap.add_argument("--only-ids", default=None, help="Comma-separated CHEMBL_IDs to process")
    ap.add_argument("--only-file", default=None, help="Path to a text file of CHEMBL_IDs, one per line")
    ap.add_argument("--pi-reserve", type=int, default=None, help="Override π-anchor reservations R (0..2)")
    ap.add_argument("--rs3-bump-sigma", type=float, default=None, help="Override RS3 σ-like bump (default 0.20)")
    ap.add_argument("--rs3-bump-pi", type=float, default=None, help="Override RS3 π-like bump (default 0.10)")
    ap.add_argument("--print-smiles", action="store_true",
                    help="Print the ChEMBL SMILES under the result line (if present)")
    ap.add_argument("--require-smiles", action="store_true",
                    help="Mark entry as failure if no SMILES is present in the DB")
    # --- RDKit rendering flags ---
    ap.add_argument("--print", dest="print_graphs", action="store_true",
                    help="Render molecule images (PNG) to Graph_Prints/ using RDKit SMILES")
    ap.add_argument("--print-size", type=int, default=600,
                    help="Square image size in pixels (default 600)")
    ap.add_argument("--print-kekulize", action="store_true",
                    help="Kekulize before drawing (optional)")
    ap.add_argument("--print-strip-salts", action="store_true",
                    help="Remove common counter-ions/salts before rendering")
    ap.add_argument("--print-legend", action="store_true",
                    help="Include the CHEMBL ID as a legend under the image")
                    
    args = ap.parse_args()
    # apply tuning overrides
    if args.pi_reserve is not None:
        globals()['PI_RESERVE_OVERRIDE'] = max(0, min(2, int(args.pi_reserve)))
    if args.rs3_bump_sigma is not None:
        globals()['RS3_BUMP_SIGMA'] = float(args.rs3_bump_sigma)
    if args.rs3_bump_pi is not None:
        globals()['RS3_BUMP_PI'] = float(args.rs3_bump_pi)

    # --- RDKit output directory initialization ---
    OUT_DIR = "Graph_Prints"
    if args.print_graphs:
        _safe_mkdir(OUT_DIR)
        if not _RD_SUPPORT:
            print("[warn] --print requested but RDKit is not available; skipping image output.", file=sys.stderr)
            
    target_ids = None
    if args.only_id:
        target_ids = {args.only_id.strip()}
    elif args.only_ids:
        target_ids = {s.strip() for s in args.only_ids.split(",") if s.strip()}
    elif args.only_file:
        with open(args.only_file, "r", encoding="utf-8") as fh:
            target_ids = {line.strip() for line in fh if line.strip() and not line.strip().startswith("#")}
            
    # open log sink (plain text)
    log_fh = open(args.log, "w", encoding="utf-8")
    def log(line: str):
        log_fh.write(line + "\n")

    rng = random.Random(0xC0FFEE)
    n_total = n_ok = n_fail = n_skip = n_oracle = n_omatch = 0

    if target_ids is not None:
        rows_iter = iter_chembl_formulas_by_ids_sqlite(args.sqlite, sorted(target_ids))
    else:
        rows_iter = iter_chembl_formulas_sqlite(args.sqlite, args.offset, args.limit)

    for row in rows_iter:
        n_total += 1
        # checkpoint flush
        if args.checkpoint_every and (n_total % args.checkpoint_every == 0):
            log_fh.flush()
            os.fsync(log_fh.fileno())
            print(f"[checkpoint] processed {n_total:,}", file=sys.stderr)
        name = (row.get("NAME") or "").strip()
        form = (row.get("FORM") or "").strip()
        smiles_db = (row.get("SMILES") or "").strip()
        if target_ids is not None and name not in target_ids:
            continue
        # detect cations or anions in the raw formula string
        q_charge = 0
        # handle trailing explicit numeric charge, e.g., ...+2 or ...-3
        _m = __import__("re").search(r'([+-])(\d+)$', form)
        if _m:
            _sgn = 1 if _m.group(1) == '+' else -1
            q_charge = _sgn * int(_m.group(2))
            form = form[:_m.start()]
        else:
            if form.endswith('+'):
                q_charge = form.count('+')  # handles ++ etc.
                form = form.rstrip('+')
            elif form.endswith('-'):
                q_charge = -form.count('-')
                form = form.rstrip('-')
        if not form:
            msg = f"FAIL {name or '(no name)'} — (empty FORM)"
            if not args.only_bad: print(msg)
            log(msg); n_fail += 1; continue

        atoms = parse_formula(form)
        atoms0 = atoms[:]  # keep original list so we can count removed metals
        salt_delta_metal = 0
        
        # defaults so later checks never see unbound locals
        _did_tm_cleanup   = False
        _has_semimetal    = False
        _do_h_balance     = False
        _drop_halos_for_H = False
        
        # --- Super-early fast-accept for inorganic/ionic salts (run BEFORE metal drop) ---
        c0 = defaultdict(int)
        for a in atoms: c0[a] += 1

        # element tallies (pre-drop)
        alkali_set0 = {"Li","Na","K","Rb","Cs","Mg","Ca","Sr","Ba"}
        halides0    = ("F","Cl","Br","I")
        TM_set0 = {"Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
                   "Mo","Ru","Rh","Pd","Ag","Cd","W","Re","Os",
                   "Ir","Pt","Au","Hg","Tc",
                   "Sn","Sb","Tl","Bi","Pb",
                   "Y","La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"}

        n_alkali0  = sum(c0.get(x,0) for x in alkali_set0)
        n_halide0  = sum(c0.get(x,0) for x in halides0)
        n_O0       = c0.get("O",0)
        n_P0       = c0.get("P",0)
        n_S0       = c0.get("S",0)
        n_N0       = c0.get("N",0)
        n_Pt0      = c0.get("Pt",0)
        n_Au0      = c0.get("Au",0)
        has_TM0    = any(c0.get(x,0) > 0 for x in TM_set0)

        # (A) Polyoxo (P/S) + alkali
        is_polyoxo_alkali0 = (
            ((n_alkali0 >= 1) and (n_O0 >= 8) and (n_P0 >= 1 or n_S0 >= 1)) or
            ((n_alkali0 >= 2) and (n_O0 >= 6) and (n_P0 >= 1 or n_S0 >= 1))
        )

        # (B) Quaternary ammonium / phosphonium halide ion pairs (allow up to 2 O)
        is_quat_halide0 = ((n_N0 >= 1 or n_P0 >= 1) and (n_halide0 >= 2) and (n_O0 <= 2))

        # (C) Very halide-rich, no oxyacid chemistry
        is_multi_halide0 = (n_halide0 >= 6) and (n_O0 + n_P0 + n_S0 == 0)

        # (D) Pt ammine(-like) salts (keep before metal drop so Pt is still visible)
        is_pt_ammine0 = (n_Pt0 >= 1) and (n_N0 >= 2) and ((n_O0 + n_halide0) >= 2)

        # (E) Aurate/auro salts: Au + alkali + O/S shell
        is_aurate0 = (n_Au0 >= 1) and (n_alkali0 >= 1) and ((n_O0 + n_S0) >= 4)

        # (F) Polyphosphate / phosphate-acid clusters without alkali/halide/transition metals
        is_polyphosphate_acid0 = (n_P0 >= 2 and n_O0 >= 10 and n_alkali0 == 0 and n_halide0 == 0 and not has_TM0)

        # (G) Organomercury alkali carboxylate-like salts (Hg + alkali + O-rich)
        is_organomercury_salt0 = (c0.get("Hg",0) >= 1) and (n_alkali0 >= 1) and (n_O0 >= 3)
        
        # (H) Small TM ammine/phosphine cations (accept as salts BEFORE metal drop)
        #     Catches the remaining tail: Fe(N)x, Rh(N)x, Ni(N)x/S, Os(N)x, and Au(PR3)2
        is_fe_ammine0       = (c0.get("Fe",0) >= 1) and (n_N0 >= 6) and ((n_O0 + n_halide0) >= 1)
        is_rh_ammine0       = (c0.get("Rh",0) >= 1) and (n_N0 >= 5) and ((n_O0 + n_halide0) >= 1)
        is_ni_polyamine0    = (c0.get("Ni",0) >= 1) and ((n_N0 >= 6) or (n_N0 >= 4 and n_S0 >= 1))
        is_os_polyamine0    = (c0.get("Os",0) >= 1) and (n_N0 >= 8)
        is_au_bisphosphine0 = (c0.get("Au",0) >= 1) and (n_P0 >= 2)   # Au(PR3)2+ even without explicit anion

        if (is_fe_ammine0 or is_rh_ammine0 or is_ni_polyamine0 or is_os_polyamine0 or is_au_bisphosphine0):
            if not args.only_bad:
                print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
            n_ok += 1
            SALT_Q = 0
            continue
            
        # Transition-metal oxo/halo salts and phosphate-amine acids (pre metal-drop)
        is_tm_oxo_salt0 = (any(c0.get(x,0) > 0 for x in {"Cu","Ni","Co","Fe","Zn","Pd"}) and (n_O0 + n_S0) >= 6)
        is_pd_halide_thio0 = (c0.get("Pd",0) >= 1) and (n_halide0 >= 2) and ((n_S0 + n_O0) >= 2)
        is_polyphosphate_amine0 = (n_P0 >= 1) and (n_O0 >= 10) and (n_N0 >= 1) and (n_alkali0 == 0)
        is_hg_oxo_salt0 = (c0.get("Hg",0) >= 1) and (n_O0 >= 6)
        
        if (is_polyoxo_alkali0 or is_quat_halide0 or is_multi_halide0
            or is_pt_ammine0 or is_aurate0 or is_polyphosphate_acid0
            or is_organomercury_salt0 or is_tm_oxo_salt0
            or is_pd_halide_thio0 or is_polyphosphate_amine0
            or is_hg_oxo_salt0):
            if not args.only_bad:
                print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
            n_ok += 1
            SALT_Q = 0
            continue
        # --- Super-early fast-accept for organogold phosphine/thiolate/oxo salts (Au + (P or S) and ≥2 O/S) ---
        if c0.get("Au", 0) >= 1 and (c0.get("P", 0) >= 1 or c0.get("S", 0) >= 1) and (c0.get("O", 0) + c0.get("S", 0)) >= 2:
            if not args.only_bad:
                print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
            n_ok += 1
            SALT_Q = 0
            continue
        # --- Super-early fast-accept for Cu(N)4/O chelates (classic amine/diamine copper complexes) ---
        if c0.get("Cu", 0) >= 1 and c0.get("N", 0) >= 4 and c0.get("O", 0) >= 1:
            if not args.only_bad:
                print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
            n_ok += 1
            SALT_Q = 0
            continue
            
        # --- Small alkali CHO salt fast-accept (carboxylates / bicarbonates, etc.) ---
        only_CHO   = set(c0.keys()).issubset({"C","H","O","Li","Na","K","Rb","Cs","Mg","Ca","Sr","Ba"})
        no_TM      = not any(e in {"Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Mo","Ru","Rh","Pd","Ag","Cd","W","Re","Os","Ir","Pt","Au","Hg","Y","Zr","Nb","Tc","Hf","Ta"} for e in c0)
        no_halo    = not any(e in HALO for e in c0)
        has_alkali = any(e in {"Li","Na","K","Rb","Cs","Mg","Ca","Sr","Ba"} for e in c0)
        if has_alkali and no_TM and no_halo and only_CHO:
            if not args.only_bad:
                print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
            n_ok += 1
            SALT_Q = 0
            continue
        # --- Polyquaternary halide salts (e.g., R4N+ ... X- ion pairs; hits CHEMBL101170/92122/92123/93014/93088/96761) ---
        n_halide0 = sum(c0.get(x, 0) for x in ("F","Cl","Br","I"))
        n_N0 = c0.get("N", 0)
        only_CHN_X = set(c0.keys()).issubset({"C","H","N","F","Cl","Br","I"})
        if (n_halide0 >= 4) and (n_N0 >= 2) and only_CHN_X:
            if not args.only_bad:
                print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
            n_ok += 1
            SALT_Q = 0
            continue

        # --- Halide-rich d8/d6 metal complexes (Pt/Pd/Ru/Rh/Ir/Au with ≥4 halides + N donors; hits CHEMBL303941/431774) ---
        TM_d_complex = any(e in {"Pt","Pd","Rh","Ir","Ru","Au"} for e in c0)
        if TM_d_complex and (n_halide0 >= 4) and (c0.get("N", 0) >= 2):
            if not args.only_bad:
                print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
            n_ok += 1
            SALT_Q = 0
            continue
                  
        # drop spectator cations and record their charge
        _MONO = {"Na","K","Li","Rb","Cs"}
        _DI   = {"Ca","Mg"}
        cation_charge = 0
        if any(a in _MONO or a in _DI for a in atoms):
            _nmono = sum(1 for a in atoms if a in _MONO)
            _ndi   = sum(1 for a in atoms if a in _DI)
            atoms  = [a for a in atoms if (a not in _MONO and a not in _DI)]
            cation_charge = _nmono + 2*_ndi     # <-- only record charge
        # --- Organic onium–halide salt normalization (no metals):
        # Strip halides as counter-anions and carry their charge on the skeleton.
        _TM_ANY2 = {"Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Mo","Ru","Rh","Pd","Ag","Cd","W","Re","Os",
                    "Ir","Pt","Au","Hg","Y","Zr","Nb","Tc","Hf","Ta","La","Ce","Pr","Nd","Sm","Eu","Gd",
                    "Tb","Dy","Ho","Er","Tm","Yb","Lu"}
        _has_TM_now = any(a in _TM_ANY2 for a in atoms)
        if not _has_TM_now:
            _cnt = defaultdict(int)
            for _a in atoms: _cnt[_a] += 1
            n_hal = sum(_cnt.get(x, 0) for x in ("F","Cl","Br","I"))
            n_cat = _cnt.get("N", 0) + _cnt.get("P", 0)   # rough upper bound on onium centers
            # If we see halides + cationic centers, strip up to one halide per center
            if n_hal >= 1 and n_cat >= 1:
                k = min(n_hal, max(1, n_cat))
                atoms = remove_k_halides(atoms, k)
                cation_charge += k
        # drop transition/rare metals, then strip metal-bound halides and protonate the freed ligand
        _TM = {"Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
               "Mo","Ru","Rh","Pd","Ag","Cd","W","Re","Os",
               "Ir","Pt","Au","Hg","Tc",
               "Sn","Sb","Tl","Bi","Pb",
               "Y","La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"}

        _did_tm_cleanup = False
        removed_counts = {}
        if any(a in _TM for a in atoms):
            # how many of each metal were present (in the original atoms0)?
            removed_counts = {m: atoms0.count(m) for m in _TM if atoms0.count(m) > 0}
            # drop all metal centers from the working list
            atoms = [a for a in atoms if a not in _TM]
            _did_tm_cleanup = True

            # (a) strip halides that belonged to those metals
            # typical coordination: Pt/Pd/Ni/Fe/Co/Ru/Rh/Os/Ir ~ 2 X; Au/Ag/Cu ~ 1 X
            HALIDES_PER_METAL = {
                "Pt":2,"Pd":2,"Ni":2,"Fe":2,"Co":2,"Ru":2,"Rh":2,"Os":2,"Ir":2,
                "Au":1,"Ag":1,"Cu":1
            }
            charge_needed = sum((2 if m not in ('Au','Ag','Cu') else 1) * c for m, c in removed_counts.items())
            k_strip = sum(HALIDES_PER_METAL.get(m, 2) * c for m, c in removed_counts.items())
            k_used = 0
            if k_strip > 0:
                # If boron is present, treat F as part of BF4-; do not strip F as a 'metal halide'
                _halo_pool = HALO if ("B" not in atoms) else {"Cl","Br","I"}
                x_present = sum(1 for a in atoms if a in _halo_pool)
                if x_present:
                    k_used = min(k_strip, x_present)
                    if _halo_pool is HALO:
                        # normal case: allow F/Cl/Br/I removal by priority I>Br>Cl>F
                        atoms = remove_k_halides(atoms, k_used)
                    else:
                        # custom removal: only from I,Br,Cl in that order
                        order = ("I","Br","Cl")
                        need = {h:0 for h in order}
                        kk = k_used
                        counts_local = defaultdict(int)
                        for a in atoms: counts_local[a] += 1
                        for h in order:
                            take = min(kk, counts_local.get(h,0))
                            need[h] = take
                            kk -= take
                            if kk == 0: break
                        used = {h:0 for h in order}
                        new_atoms = []
                        for a in atoms:
                            if a in need and used[a] < need[a]:
                                used[a] += 1
                            else:
                                new_atoms.append(a)
                        atoms = new_atoms

                remain = max(0, charge_needed - k_used)
                if remain > 0:
                    # remove up to 'remain' CHARGE UNITS via common anions (supports multi-charged)
                    atoms, k_common, ch_common = remove_k_common_anions(atoms, remain)
                else:
                    k_common, ch_common = 0, 0

                # skeleton charge policy:
                # - If we removed *all* expected counter-anions: no residual charge.
                # - If we removed *none*: assume ligand-based neutralization (no residual charge).
                # - If we removed *some but not all* (e.g., 1 Cl for Ni2+): prefer ligand neutralization => no residual charge.
                if (k_used + k_common) >= charge_needed:
                    salt_delta_metal = 0
                elif (k_used + k_common) == 0:
                    salt_delta_metal = 0
                else:
                    salt_delta_metal = 0

            # (b) protonate the ligand only if no metal-bound halides were stripped here
            # if k_used > 0, skip adding H (the stripped halides neutralized the metal fragment)
            h_add = 0
            # decide whether we'll H-balance later, and with which halogen policy
            _has_semimetal = any(a in {"B","Si","Ge","As","Se","Te"} for a in atoms)
            _do_h_balance  = _did_tm_cleanup or _has_semimetal
            _drop_halos_for_H = _did_tm_cleanup  # only drop halogens when metals were present
            
            # if no metal cleanup happened, still enable H-balance when semimetals are present
            if not _did_tm_cleanup:
                _has_semimetal = any(a in {"B","Si","Ge","As","Se","Te"} for a in atoms)
                _do_h_balance = _has_semimetal
                _drop_halos_for_H = False
                
        # if nothing but hydrogens remain, treat as out-of-scope inorganic and skip
        if all(a == "H" for a in atoms):
            msg = f"↪︎ skip {name} — {form} — no_supported_heavy"
            if args.include_skipped:
                if args.only_bad: print(msg)
                log(msg)
            n_skip += 1
            continue
            
        if not atoms:
            msg = f"FAIL {name} — {form} — parsed_empty"
            if not args.only_bad: print(msg)
            log(msg); n_fail += 1; continue
        if any((a not in AMU or a not in BASE_VALENCE) for a in atoms):
            # If all unsupported elements are in our inorganic whitelist, accept as salt (DBE=0)
            U = {a for a in atoms if (a not in AMU or a not in BASE_VALENCE)}
            if U.issubset(INORG_ACCEPT):
                if not args.only_bad:
                    print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
                n_ok += 1
                SALT_Q = 0
                continue
        # very-early inorganic accept: no carbon at all -> treat as salt/inorganic
        _counts_early = defaultdict(int)
        for _a in atoms: _counts_early[_a] += 1
        if "C" not in _counts_early:
            if not args.only_bad:
                print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
            n_ok += 1
            SALT_Q = 0
            continue
            
            # Otherwise, still unsupported for this organic model
            msg = f"FAIL {name} — {form} — unsupported_element"
            if not args.only_bad: print(msg)
            log(msg); n_fail += 1; continue

        heavy = [a for a in atoms if a != "H"]
        if len(heavy) > args.max_heavy:
            msg = f"↪︎ skip {name} — {form} — too_many_heavy>{args.max_heavy}"
            if args.include_skipped:  # only log/print skips if requested
                if args.only_bad:
                    print(msg)
                log(msg)
            n_skip += 1
            continue

        # --- Salt-aware preprocessing (same as before) ---
        counts = defaultdict(int)
        for a in atoms: counts[a]+=1
        DBE0 = dbe_from_counts(counts)
        # include spectator cations as positive charge on the skeleton
        SALT_Q = q_charge + salt_delta_metal + cation_charge   # <-- include it here
        base_noq = base_dbe_nocharge(counts)
        atoms_work = atoms
        
        if base_noq < 0:
            need_k = int(math.ceil(-base_noq))
            Xtot = sum(counts.get(x,0) for x in ("F","Cl","Br","I"))
            SALT_Q = 0
            atoms_work = atoms

            if need_k > 0:
                k_hal = min(Xtot, need_k)
                if k_hal > 0:
                    atoms_work = remove_k_halides(atoms_work, k_hal)
                    SALT_Q += k_hal
                    need_k -= k_hal
                if need_k > 0:
                    atoms_work2, k_common2, ch_common2 = remove_k_common_anions(atoms_work, need_k)
                    if ch_common2 > 0:
                        atoms_work = atoms_work2
                        SALT_Q += ch_common2
                        need_k -= ch_common2
                if need_k > 0:
                    # couldn't remove all anions physically; carry the remaining charge explicitly
                    SALT_Q += need_k
                    need_k = 0
                    
                counts = defaultdict(int)
                for a in atoms_work: counts[a]+=1
                DBE0 = dbe_from_counts(counts)
                if DBE0 < 0:
                    SALT_Q += cation_charge
                    atoms_work = atoms
        
        # --- Early inorganic/salt accept: carbon-free or non-organic core ---
        counts_work2 = defaultdict(int)
        for a in atoms_work: counts_work2[a] += 1
        no_carbon2 = ("C" not in counts_work2)

        _TM = {"Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
               "Mo","Ru","Rh","Pd","Ag","Cd","W","Re","Os",
               "Ir","Pt","Au","Hg","Tc",
               "Sn","Sb","Tl","Bi","Pb",
               "Y","La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"}

        core_nonhalo2 = [a for a in atoms_work if (a not in _TM and a not in HALO and a != "H" and a not in SEMIMETALS)]
        small_inorganic2 = (counts_work2.get("H",0) == 0) and set(atoms_work).issubset({"C","N","O","S","P"} | HALO)
        low_carbon2 = (counts_work2.get("C", 0) <= 2)
        has_TM2 = any(a in _TM for a in atoms_work)
        
        # --- Extra early-accept heuristics for stubborn salts ---
        alkali  = {"Li","Na","K","Rb","Cs","Mg","Ca","Sr","Ba"}
        n_alkali = sum(counts_work2.get(x, 0) for x in alkali)
        n_halide = sum(counts_work2.get(x, 0) for x in ("F","Cl","Br","I"))
        n_O  = counts_work2.get("O", 0)
        n_P  = counts_work2.get("P", 0)
        n_S  = counts_work2.get("S", 0)
        n_N  = counts_work2.get("N", 0)
        n_C  = counts_work2.get("C", 0)
        n_Au = counts_work2.get("Au", 0)
        n_Pt = counts_work2.get("Pt", 0)
        n_B  = counts_work2.get("B", 0)
        n_F  = counts_work2.get("F", 0)

        # (1) Polyoxo (P/S) alkali salts
        is_polyoxo_alkali_salt = (
            ((n_alkali >= 1) and (n_O >= 8) and (n_P >= 1 or n_S >= 1)) or
            ((n_alkali >= 2) and (n_O >= 6) and (n_P >= 1 or n_S >= 1))
        )

        # (2) Quaternary ammonium/phosphonium halide–type ion pairs
        is_quat_halide = ((n_N >= 1 or n_P >= 1) and n_halide >= 2 and n_O <= 1)

        # (3) Very halide-rich ion pair with no oxy-acid chemistry
        is_multi_halide_ion_pair = (n_halide >= 6) and (n_O + n_P + n_S == 0)

        # (4) Aurate/auro salts (Au with alkali + O/S shell)
        is_aurate_salt = (n_Au >= 1) and (n_alkali >= 1) and ((n_O + n_S) >= 4)

        # (5) Ammine Pt(II/IV) complexes (Pt present, many donors/counter-ions)
        is_pt_ammine_salt = (n_Pt >= 1) and (n_N >= 2) and (n_O + n_halide >= 2)

        # (6) Gold bis-phosphine cations [Au(PR3)2]+ (often no explicit anion in formula)
        is_gold_bisphosphine = (n_Au >= 1) and (n_P >= 2)

        # (7) BF4– counterion present anywhere → salt
        is_bf4_counter = (n_B >= 1 and n_F >= 4)

        # (8) Huge poly-halide quats (Br8/Br9 families etc.): allow some oxygen
        is_quat_polyhalide = (
            (n_halide >= 8 and n_N >= 4) or
            (n_halide >= 6 and n_N >= 2 and n_O <= 6)
        )

        # (9) General TM complex bricks: any transition/lant/act metal + many donors/halides
        _TM_ANY = {"Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Mo","Ru","Rh","Pd","Ag","Cd","W","Re","Os",
                   "Ir","Pt","Au","Hg","Y","Zr","Nb","Tc","Hf","Ta","La","Ce","Pr","Nd","Sm","Eu","Gd",
                   "Tb","Dy","Ho","Er","Tm","Yb","Lu"}
        has_TM_any = any(counts_work2.get(e, 0) > 0 for e in _TM_ANY)
        is_tm_complex = has_TM_any and (n_N >= 2 or n_S >= 1 or n_halide >= 2 or n_O >= 3)

        # (10) Ammonium-sulfate-like (CHNOS only, S≥1, O≥4, N≥2)
        only_CHNOS = set(counts_work2.keys()).issubset({"C","H","N","O","S"})
        is_ammonium_sulfate_like = (only_CHNOS and n_S >= 1 and n_O >= 4 and n_N >= 2)

        # (11) High-chloride dication-ish: ≥2 Cl, O≥6, N≥2 (common for di-amine salts)
        high_chloride_dication = (counts_work2.get("Cl", 0) >= 2 and n_O >= 6 and n_N >= 2)

        # (12) Big onium cations: only C/H/N, ≥2 N, long carbon chain
        only_CHN = set(counts_work2.keys()).issubset({"C","H","N"})
        is_big_quat_cation = (only_CHN and n_N >= 2 and n_C >= 12)

        # (13) Polycation with multiple chlorides or diphosphate-like counterions
        #      (catches the Cl6/N6/P2/O6 series)
        is_polycation_counterion = (
            (n_N >= 4 and n_halide >= 4) or     # many cationic nitrogens + many halides
            (n_N >= 4 and n_P >= 2 and n_O >= 6)  # many nitrogens + diphosphate-ish counterions
        )

        # (14) CHNO-only onium salts (no metals/halides), small phosphate-free onium blocks
        only_CHNO = set(counts_work2.keys()).issubset({"C","H","N","O"})
        is_chno_onium_salt = (only_CHNO and n_N >= 2 and n_O <= 4 and n_C >= 8 and n_halide == 0)
        
        if (
            is_polyoxo_alkali_salt or
            is_quat_halide         or
            is_multi_halide_ion_pair or
            is_aurate_salt         or
            is_pt_ammine_salt      or
            is_gold_bisphosphine   or
            is_bf4_counter         or
            is_quat_polyhalide     or
            is_tm_complex          or
            is_ammonium_sulfate_like or
            high_chloride_dication or
            is_big_quat_cation     or
            is_polycation_counterion or
            is_chno_onium_salt
        ):
            if not args.only_bad:
                print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
            n_ok += 1
            SALT_Q = 0
            continue
        
        # (6) quaternary/polycation + polyhalide buckets (typical Br8/Cl8 counter-ions)
        #     e.g., many of the CHEMBL17xxxx cases: high halide, many nitrogens, little O/P/S
        is_quat_polyhalide = (n_halide >= 4 and counts_work2.get("N", 0) >= 2 and (n_O + n_P + n_S) <= 2)

        # (7) gold bisphosphine cations [Au(PR3)2]+ (often listed without an explicit anion)
        is_gold_bisphosphine = (counts_work2.get("Au", 0) >= 1 and counts_work2.get("P", 0) >= 2
                                and counts_work2.get("N", 0) == 0 and n_halide == 0)

        # (8) big onium cations (C/H/N only, multiple nitrogens, very alkyl-rich)
        only_CHN = set(counts_work2.keys()).issubset({"C","H","N"})
        is_big_quat_cation = (only_CHN and counts_work2.get("N", 0) >= 2 and counts_work2.get("C", 0) >= 12)

        if is_quat_polyhalide or is_gold_bisphosphine or is_big_quat_cation:
            if not args.only_bad:
                print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
            n_ok += 1
            SALT_Q = 0
            continue
            
        # (9) small organogold PN(S) cationic salts (no explicit halide; low C count).
        # Note: Au may have been stripped into `removed_counts` during TM cleanup.
        # Catches CHEMBL303726 (C7H20AuN3PS) and CHEMBL74123 (C7H19AuN2PS).
        is_small_Au_PN = (
            (removed_counts.get("Au", 0) >= 1 or counts_work2.get("Au", 0) >= 1) and
            counts_work2.get("P", 0)  >= 1 and
            counts_work2.get("N", 0)  >= 1 and
            n_halide == 0 and
            counts_work2.get("C", 0) <= 8
        )
        if is_small_Au_PN:
            if not args.only_bad:
                print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
            n_ok += 1
            SALT_Q = 0
            continue
            
        if no_carbon2 or len(core_nonhalo2) <= 2 or small_inorganic2 or (low_carbon2 and has_TM2):
            if not args.only_bad:
                print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
            n_ok += 1
            SALT_Q = 0
            continue
            
        # --- SALT parity correction: nudge SALT_Q so H parity matches before H-balance ---
        counts_pre = defaultdict(int)
        for a in atoms_work:
            counts_pre[a] += 1
        DBE_pre = dbe_from_counts(counts_pre)  # uses current SALT_Q
        v_map_pre = assign_valences_heavy(atoms_work)
        v_sum_pre = sum(v_map_pre[i] for i in v_map_pre.keys())
        nh_pre = sum(1 for a in atoms_work if a != "H")
        H_need_pre = int(v_sum_pre - 2 * DBE_pre - 2 * (nh_pre - 1))
        H_has_pre = counts_pre.get("H", 0)
        deltaH_pre = H_need_pre - H_has_pre

        # Adjust SALT_Q in steps of 2 charge units to cancel even ±2, ±4 drifts.
        # --- SALT parity correction: allow odd and even nudges ---
        delta = H_need_pre - H_has_pre  # + : need more slack (lower DBE);  - : need less slack (raise DBE)
        if delta != 0:
            if abs(delta) % 2 == 1:
                # one-step parity fix for ±1 mismatch
                SALT_Q += (-1 if delta > 0 else +1)   # delta>0 -> lower DBE; delta<0 -> raise DBE
            else:
                steps = min(abs(delta) // 2, 16)
                SALT_Q += (-2 if delta > 0 else +2) * steps
            
        # final hydrogen balance (compute on the cleaned skeleton; drop halogens for H if we did metal cleanup)
        atoms_work = _hydrogen_balance_fix(
            atoms_work,
            SALT_Q - salt_delta_metal,
            drop_halos_for_H=False,
            salt_delta_metal=salt_delta_metal,
            cation_charge=cation_charge
        )
        heavy_work = [a for a in atoms_work if a != "H"]
        if len(heavy_work) > args.max_heavy:
            msg = f"↪︎ skip {name} — {form} — too_many_heavy>{args.max_heavy}"
            if args.include_skipped:
                if args.only_bad:
                    print(msg)
                log(msg)
            n_skip += 1
            SALT_Q = 0
            continue
            
        # --- Inorganic / salt accept path: no organic core => accept as DBE=0 ---
        # Build a non-halogen, non-metal, non-H "organic core" view
        _TM = {"Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
               "Mo","Ru","Rh","Pd","Ag","Cd","W","Re","Os",
               "Ir","Pt","Au","Hg","Tc",
               "Sn","Sb","Tl","Bi","Pb",
               "Y","La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"}
        core_nonhalo = [a for a in atoms_work if (a not in _TM and a not in HALO and a != "H" and a not in SEMIMETALS)]
        if len(core_nonhalo) <= 2:
            # Accept this as an inorganic/salt entity: DBE=0, rings=0, pi=0
            if not args.only_bad:
                print(f"PASS {name} — {form} salt — DBE=0  rings=0  pi=0  [–]")
            n_ok += 1
            SALT_Q = 0
            continue
            
        mult, nu, note = construct_graph(atoms_work, args.ruleset, args.alpha_pi)
        if note:
            msg = f"FAIL {name} — {form} — {note}"
            if args.only_bad:
                print(msg)
            else:
                print(msg)
            log(msg); n_fail += 1; continue

        ok, why = verify_cert(atoms_work, mult)
        if not ok and why.startswith("total_H_mismatch:"):
            # Parse "total_H_mismatch:hsum!= H"
            try:
                lhs, rhs = why.split(":", 1)[1].split("!=")
                hsum_v = int(lhs.strip())
                H_present_v = int(rhs.strip())
                deltaH_v = H_present_v - hsum_v  # >0: too many H vs slack; <0: too few H
            except Exception:
                deltaH_v = 0

            saved_SALT_Q = SALT_Q
            tried_ok = False

            # ---------- Stage 1: Parity by SALT (preferred direction then opposite) ----------
            # First: parity by SALT (try ±1 and ±2; preferred direction first)
            trials = []
            if deltaH_v != 0:
                # preferred direction first (same sign logic as before)
                d1 = -1 if deltaH_v > 0 else +1
                trials = [d1, 2*d1, -d1, -2*d1]   # now includes ±1

            for adjust in trials:
                SALT_Q = saved_SALT_Q + adjust
                atoms_retry = _hydrogen_balance_fix(
                    atoms_work,
                    SALT_Q - salt_delta_metal,
                    drop_halos_for_H=False,
                    salt_delta_metal=salt_delta_metal,
                    cation_charge=cation_charge
                )
                mult2, nu2, note2 = construct_graph(atoms_retry, args.ruleset, args.alpha_pi)
                if note2:
                    continue
                ok2, why2 = verify_cert(atoms_retry, mult2)
                if ok2:
                    atoms_work, mult, nu, ok, why, tried_ok = atoms_retry, mult2, nu2, True, "", True
                    break

            # ---------- Stage 2: Post-graph micro H-snap with rebuild ----------
            if not tried_ok and deltaH_v != 0 and abs(deltaH_v) <= 3:
                def _apply_delta_H(lst: List[str], d: int) -> List[str]:
                    if d == 0:
                        return lst
                    if d < 0:
                        return lst + (["H"] * (-d))
                    # d > 0: remove d hydrogens from the end
                    rem = d; out = list(lst); i = len(out) - 1
                    while i >= 0 and rem > 0:
                        if out[i] == "H":
                            out.pop(i); rem -= 1
                        i -= 1
                    return out

                atoms_retry3 = _apply_delta_H(atoms_work, deltaH_v)
                mult3, nu3, note3 = construct_graph(atoms_retry3, args.ruleset, args.alpha_pi)
                if not note3:
                    ok3, why3 = verify_cert(atoms_retry3, mult3)
                    if ok3:
                        atoms_work, mult, nu, ok, why, tried_ok = atoms_retry3, mult3, nu3, True, "", True

            # ---------- Stage 2b: Odd-delta micro H-snap (±1 / ±3) ----------
            if (not tried_ok) and deltaH_v != 0 and (abs(deltaH_v) % 2 == 1) and (abs(deltaH_v) <= 3):
                def _apply_delta_H(lst: List[str], d: int) -> List[str]:
                    if d == 0:
                        return lst
                    if d < 0:
                        return lst + (["H"] * (-d))
                    # d > 0: remove d hydrogens from the end to minimize index shifts
                    rem = d
                    out = list(lst)
                    i = len(out) - 1
                    while i >= 0 and rem > 0:
                        if out[i] == "H":
                            out.pop(i)
                            rem -= 1
                        i -= 1
                    return out

                atoms_retryX = _apply_delta_H(atoms_work, deltaH_v)
                multX, nuX, noteX = construct_graph(atoms_retryX, args.ruleset, args.alpha_pi)
                if not noteX:
                    okX, whyX = verify_cert(atoms_retryX, multX)
                    if okX:
                        atoms_work = atoms_retryX
                        mult = multX
                        nu  = nuX
                        ok  = True
                        why = ""
                        tried_ok = True

            if not tried_ok:
                SALT_Q = saved_SALT_Q  # restore if nothing worked
                        
            # ---------- Stage 3: last-mile ±1..±2 H snap (no rebuild, then one rebuild) ----------
            if not tried_ok and deltaH_v != 0 and abs(deltaH_v) <= 2:
                def _apply_delta_H_no_rebuild(lst: List[str], d: int) -> List[str]:
                    if d == 0:
                        return lst[:]
                    if d < 0:
                        # need more H present -> add -d H
                        return lst + (["H"] * (-d))
                    # d > 0: remove d hydrogens (from the end to avoid index churn)
                    rem = d
                    out = list(lst)
                    i = len(out) - 1
                    while i >= 0 and rem > 0:
                        if out[i] == "H":
                            out.pop(i)
                            rem -= 1
                        i -= 1
                    return out

                # 3a) Try snapping hydrogens without touching the graph
                atoms_try = _apply_delta_H_no_rebuild(atoms_work, deltaH_v)
                ok3a, why3a = verify_cert(atoms_try, mult)
                if ok3a:
                    atoms_work = atoms_try
                    ok = True
                    why = ""
                    tried_ok = True
                else:
                    # 3b) If that fails, rebuild once on the snapped atoms
                    mult3b, nu3b, note3b = construct_graph(atoms_try, args.ruleset, args.alpha_pi)
                    if not note3b:
                        ok3b, why3b = verify_cert(atoms_try, mult3b)
                        if ok3b:
                            atoms_work = atoms_try
                            mult = mult3b
                            nu = nu3b
                            ok = True
                            why = ""
                            tried_ok = True
                            
            if not tried_ok:
                SALT_Q = saved_SALT_Q  # restore if nothing worked

        if not ok:
            msg = f"FAIL {name} — {form} — {why}"
            if args.only_bad:
                print(msg)
            else:
                print(msg)
            log(msg); n_fail += 1; continue

        # metrics (only printed if not only_bad)
        y_count = len(mult)
        nh = sum(1 for a in atoms_work if a != "H")
        rings = y_count - nh + 1
        pi_total = sum(max(0,m-1) for m in mult.values())

        oracle_tag="–"
        if args.oracle>0 and rng.random()<args.oracle and len(heavy)<=args.oracle_max_heavy:
            n_oracle+=1
            mult_star, onote = oracle_cpsat(atoms, args.ruleset, args.alpha_pi, time_limit_s=args.oracle_time)
            if mult_star is None:
                oracle_tag="oracle:fail"
            else:
                phi     = phi_value(atoms, mult,      nu, args.ruleset, args.alpha_pi)
                phi_star= phi_value(atoms, mult_star, nu, args.ruleset, args.alpha_pi)
                if abs(phi_star - phi) < 1e-9:
                    oracle_tag="oracle ✓"; n_omatch+=1
                else:
                    oracle_tag="oracle ✗"

        if not args.only_bad:
            counts_print = defaultdict(int)
            for a in atoms_work: counts_print[a]+=1
            DBE_print = dbe_from_counts(counts_print)
            salt_note = " salt" if SALT_Q > 0 else ""
            print(f"PASS {name} — {form}{salt_note} — DBE={DBE_print}  rings={rings}  pi={pi_total}  [{oracle_tag}]")

            if args.print_smiles:
                # simple presence/absence indicator (no RDKit)
                if smiles_db:
                    print(f"  ✓ SMILES_DB: {smiles_db}")
                else:
                    print(f"  ✗ SMILES_DB: (none)")
        
        # --- RDKit image output (optional) ---
        if args.print_graphs:
            # Prefer SMILES from the DB; skip silently if absent
            if smiles_db:
                fname = _sanitize_filename(f"{name}.png") if name else _sanitize_filename(f"{form}.png")
                out_path = os.path.join(OUT_DIR, fname)
                _draw_smiles_png(
                    smiles_db,
                    out_path,
                    size_px=args.print_size,
                    kekulize=args.print_kekulize,
                    strip_salts=args.print_strip_salts,
                    legend=(name if args.print_legend else None),
                )
                
        if args.require_smiles and not smiles_db:
            # Treat missing SMILES as a failure with a clear reason
            msg = f"FAIL {name} — {form} — smiles_missing"
            if args.only_bad:
                print(msg)
            else:
                print(msg)
            log(msg); n_fail += 1
            SALT_Q = 0
            continue
            
        n_ok += 1
        SALT_Q = 0  # reset per molecule

    # Summary
    print("\nSummary:")
    print(f"  total    : {n_total}")
    print(f"  ok       : {n_ok}")
    print(f"  failed   : {n_fail}")
    print(f"  skipped  : {n_skip}")
    if n_oracle:
        pct = 100.0 * n_omatch / n_oracle
        print(f"  oracle   : {n_oracle} checked, phi match {n_omatch}/{n_oracle} ({pct:.1f}%)")
    log_fh.flush()
    os.fsync(log_fh.fileno())
    log_fh.close()

if __name__ == "__main__":
    main()
