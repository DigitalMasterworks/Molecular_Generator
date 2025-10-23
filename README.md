# Molecular_Generator

A fully deterministic, geometry-free pipeline for mapping molecular formulas to unique, valence-justified molecular graphs—anchored by physics, integer chemistry, and auditable algorithms.

---

## Overview

**Molecular_Generator** is an open-source toolkit for:
- Registry-scale molecular formula validation (ChEMBL-36)
- Registry-novel molecule enumeration (stars-and-bars, {C,N,O,S,P})
- Deterministic, auditable conversion from formula to unique connectivity
- Reproducible physical constants, lepton/atomic ladder, and integer invariants

There is **no 3D geometry, machine learning, or substructure heuristics**—every output is integer-checked, explicit, and fully reproducible.

---

## Program Structure

### Core Components
- **Formula Parsing:** Reads raw formula strings, extracts element counts and explicit charge.
- **Rung Descriptor:** Maps atomic mass to a continuous scale anchored by the electron and muon.
- **Kernel & Weights:** Assigns pairwise affinities via a dual-peak kernel, using only rung separation.
- **Bond Order Caps & Integer Rules:** Applies deterministic caps for each element pair; all arithmetic is integer-checked.
- **Salt & Charge Normalization:** Removes spectator ions, handles registry salts, and computes net skeleton charge.
- **Graph Construction:** Builds a degree-constrained tree, attaches halogens as leaves, reserves headroom for multiple bonds.
- **Unsaturation Assignment:** Spends DBE as cycles or $\pi$ bonds, using kernel weights.
- **Verification:** Enforces five integer-based checks for chemical plausibility and DBE consistency.
- **SMILES Output:** Converts connectivity to canonical SMILES (via RDKit or equivalent).

---

## Experiments and Theoretical Foundations

### 1. Muon Mass and Ladder Calibration

- **Goal:** Anchor the rung descriptor using *only* the electron and muon mass.
- **Approach:** 
    - Fix rung $k=3$ at the electron mass, $k=16$ at the muon mass.
    - Calibrate the scale $g = (m_\mu / m_e)^{1/13}$.
- **Result:** Enables mapping of all atomic masses onto a reproducible, physically grounded ladder.

### 2. Lepton and Atomic Mass Findings

- **Goal:** Demonstrate how lepton anchors generate a continuous atomic coordinate.
- **Approach:** 
    - Use standard atomic weights and lepton calibration to produce rung values for all elements.
    - Validate ladder’s physical meaning by comparison with atomic/periodic trends.
- **Result:** All element-specific chemistry downstream is grounded in a scale that is timeless and auditable.

### 3. Bond Construction and Sigma/π Matching

- **Goal:** Provide an exact, parameter-free mapping from composition to possible bonds.
- **Approach:**
    - Use the rung-based kernel to rank all possible atom pairs.
    - Apply deterministic bond order caps, degree constraints, and tree construction (Kruskal style).
    - Allocate DBE as cycles or $\pi$ increments using kernel weights, not heuristics.
- **Result:** Every structure is integer-justified, unique, and verifiable without geometry.

### 4. Molecule Enumeration and Registry Validation

- **Registry Validation:**  
    - **Input:** All 2.85 million in-scope ChEMBL-36 formulas.
    - **Process:** Deterministic connectivity construction, SMILES comparison to registry.
    - **Performance:** Completed in under two hours on a single workstation, zero geometry calls, zero failures.

- **Novel Enumeration:**  
    - **Input:** All $\{\mathrm{C,N,O,S,P}\}$ formulas, 2–18 heavy atoms (3,162,509 candidates).
    - **Process:** Filter out ChEMBL matches; apply deterministic construction and stability screening.
    - **Result:** 421,928 registry-novel candidates, 69,307 stable by RDKit filter.
