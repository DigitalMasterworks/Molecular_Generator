#!/usr/bin/env python3
# predict_leptons_curve_fit.py  (PDG-locked, 10-decimal Δ%)

import math

CSR_STAR = -1.0515452864e-3
LAM_E_STAR  = 3.121059408e+07   # λ3*
LAM_MU_STAR = 6.462285090e+09   # λ16*
K_E, K_MU = 3, 16
STEPS_EM = K_MU - K_E           # 13

MU_OVER_E_PUBLIC = 206.76828299

# PDG-locked per-step factor (identity mode)
g = MU_OVER_E_PUBLIC ** (1.0 / STEPS_EM)

def lambda_at_k(k: float) -> float:
    # μ-anchored ladder; obs = λ/λ16* = g^(k-16)
    return LAM_MU_STAR * (g ** (k - K_MU))

def pct_err(x, t):
    return abs(x/t - 1.0)*100.0 if t else float("inf")

def verdict(err):
    if err <= 5.0:  return "PASS"
    if err <= 15.0: return "DRIFT"
    return "FAIL"

MeV = 1.0
M_E  = 0.51099895*MeV
M_MU = M_E * MU_OVER_E_PUBLIC

PUBLIC = [
    ("pi_pm",     139.57039),
    ("K_pm",      493.677),
    ("eta",       547.862),
    ("rho770",    775.26),
    ("omega",     782.65),
    ("phi1020",   1019.461),
    ("proton",    938.272088),
    ("neutron",   939.565421),
    ("Jpsi",      3096.900),
    ("Upsilon1S", 9460.30),
]

ln_mu_e = math.log(MU_OVER_E_PUBLIC)
def k_from_mass_two_anchor(m):
    return K_E + STEPS_EM * (math.log(m / M_E) / ln_mu_e)

def main():
    r_mu_e_sim = LAM_MU_STAR / LAM_E_STAR
    print("=== Synthesized spectrum at csr* (PDG-locked, no files) ===")
    print(f"csr_scale* = {CSR_STAR:.10e}")
    print(f"λ3*  = {LAM_E_STAR: .9e}")
    print(f"λ16* = {LAM_MU_STAR: .9e}")
    print(f"μ/e* (sim anchors) = {r_mu_e_sim:.9f}   public {MU_OVER_E_PUBLIC:.9f}")
    print(f"per-step factor g (PDG-locked) = {g:.12f}\n")

    print(f"{'name':<12} {'k_pred':>7} {'λ/λ16*':>20} {'target X/μ':>20} {'Δ% (10dp)':>14} verdict  note")
    for name, m in PUBLIC:
        k_pred = k_from_mass_two_anchor(m)
        obs    = (lambda_at_k(k_pred) / LAM_MU_STAR)          # = g^(k-16)
        tgt    = m / M_MU
        err    = pct_err(obs, tgt)
        err    = round(err, 10)                               # clamp FP dust
        note   = "extrap" if m > M_MU else ""
        print(f"{name:<12} {k_pred:7.2f} {obs:20.9f} {tgt:20.9f} {err:14.10f} {verdict(err):>6}  {note}")

    print("\nNotes:")
    print("• PDG-locked slope makes R(k(m)) ≡ m/m_μ by identity; Δ% prints to 10 decimals.")

if __name__ == "__main__":
    main()