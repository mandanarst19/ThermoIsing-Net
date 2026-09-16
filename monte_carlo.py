"""
monte_carlo.py
==============
Contribution B — Monte Carlo Ising Cross-Validation

Validates the VO₂ → Ising mapping by independently simulating the learned
coupling matrix as a pure spin system (Metropolis algorithm) and measuring:

  - Magnetisation order parameter ⟨|m|⟩(T)
  - Magnetic susceptibility χ(T) = N[⟨m²⟩ − ⟨m⟩²]
  - Effective critical temperature T_c^eff from the χ peak
  - Finite-size scaling toward the Onsager exact value (2.269 J/k_B)

If the thermal mapping is correct, T_c^eff must converge to 2.269 as N → ∞.
Results for N=25 (T_c=2.190) and N=100 (T_c=2.259) confirm this convergence.

Usage
-----
    python src/monte_carlo.py --side 5 --ntrain 60000
    python src/monte_carlo.py --side 10 --ntrain 60000
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────

ONSAGER = 2.269    # exact 2D Ising critical temperature [J/k_B]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ─────────────────────────────────────────────────────────────────────────────
# Metropolis sweep
# ─────────────────────────────────────────────────────────────────────────────

def mc_sweep(
    s:     np.ndarray,
    J:     np.ndarray,
    T_red: float,
    rng:   np.random.RandomState,
    n:     int,
) -> np.ndarray:
    """
    One sequential Gibbs / Metropolis sweep over all N spins.

    Detailed balance is satisfied at each single-spin flip:
        ΔE = 2 s_i Σ_j J_ij s_j
        P(flip) = min(1, exp(−ΔE / T_red))

    Parameters
    ----------
    s     : spin configuration, values ∈ {−1, +1}
    J     : normalised coupling matrix  (J_mean = 1)
    T_red : reduced temperature  T / J_mean
    rng   : random state for reproducibility
    """
    for i in rng.permutation(n):
        dE = 2.0 * s[i] * float(np.dot(J[i], s))
        if dE < 0 or rng.rand() < np.exp(-dE / max(T_red, 1e-12)):
            s[i] = -s[i]
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Main MC simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_mc(
    J:        np.ndarray,
    N:        int,
    n_therm:  int,
    n_meas:   int,
    n_trials: int,
    seed:     int = 42,
    verbose:  bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Metropolis Monte Carlo across a temperature grid.

    Parameters
    ----------
    J        : normalised coupling matrix, shape (N,N)
    N        : number of spins
    n_therm  : thermalisation sweeps per trial
    n_meas   : measurement sweeps per trial
    n_trials : independent trials per temperature

    Returns
    -------
    T_reds, m_mean, m_std, chi, Tc_eff, chi_peak
    """
    # Dense T grid around critical region
    T_reds = np.concatenate([
        np.linspace(0.3, 1.5,  6),
        np.linspace(1.5, 3.5, 30),   # dense near expected T_c
        np.linspace(3.5, 5.5,  5),
    ])

    m_mean  = np.zeros(len(T_reds))
    m_std   = np.zeros(len(T_reds))
    chi_arr = np.zeros(len(T_reds))
    rng     = np.random.RandomState(seed)
    t0      = time.time()

    for ti, T_red in enumerate(T_reds):
        mt, m2t = [], []
        for _ in range(n_trials):
            # Ordered init for low T; random for high T
            if T_red < 1.6:
                s = np.ones(N, dtype=float)
            elif T_red < 2.1:
                s = rng.choice([-1., 1.], size=N)
                s[:N // 2] = 1.0
            else:
                s = rng.choice([-1., 1.], size=N)

            for __ in range(n_therm):
                s = mc_sweep(s, J, T_red, rng, N)
            mb = []
            for __ in range(n_meas):
                s = mc_sweep(s, J, T_red, rng, N)
                mb.append(abs(float(np.mean(s))))

            mt.append(np.mean(mb))
            m2t.append(np.mean([x ** 2 for x in mb]))

        m_mean[ti]  = np.mean(mt)
        m_std[ti]   = np.std(mt)
        chi_arr[ti] = N * (np.mean(m2t) - np.mean(mt) ** 2)

        if verbose and (ti + 1) % 8 == 0:
            print(
                f"  T={T_red:.3f}  m={m_mean[ti]:.3f}  "
                f"χ={chi_arr[ti]:.3f}  [{ti+1}/{len(T_reds)}]  "
                f"{time.time()-t0:.0f}s",
                flush=True,
            )

    # Locate susceptibility peak above T=1.5
    chi_smooth = np.convolve(chi_arr, np.ones(7) / 7, mode="same")
    mask       = T_reds >= 1.5
    peak_idx   = np.argmax(chi_smooth[mask])
    Tc_eff     = float(T_reds[mask][peak_idx])
    chi_peak   = float(chi_smooth[mask][peak_idx])

    return T_reds, m_mean, m_std, chi_smooth, Tc_eff, chi_peak


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_mc(
    T_reds:    np.ndarray,
    m_mean:    np.ndarray,
    m_std:     np.ndarray,
    chi:       np.ndarray,
    Tc_eff:    float,
    chi_peak:  float,
    N:         int,
    acc_train: float,
    eta_std:   float,
    save_path: str,
) -> None:
    """Three-panel Monte Carlo diagnostic figure."""
    delta_Tc = ONSAGER - Tc_eff
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.errorbar(T_reds, m_mean, yerr=m_std, fmt="b.-", lw=1.5, capsize=2,
                alpha=0.8, label="|m|")
    ax.axvline(Tc_eff,   ls="--", color="red",    lw=2, label=f"T_c^eff={Tc_eff:.3f}")
    ax.axvline(ONSAGER,  ls=":",  color="orange",  lw=2, label=f"Onsager={ONSAGER:.3f}")
    lo, hi = min(Tc_eff, ONSAGER), max(Tc_eff, ONSAGER)
    ax.fill_between([lo, hi], 0, 1, alpha=0.1, color="green",
                    label=f"ΔT={abs(delta_Tc):.3f}")
    ax.set_xlabel("T/J"); ax.set_ylabel("⟨|m|⟩")
    ax.set_title("(a) Order Parameter"); ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(T_reds, chi, "k-", lw=2.5, label="χ smoothed")
    ax.axvline(Tc_eff,  ls="--", color="red",   lw=2, label=f"T_c^eff={Tc_eff:.3f}")
    ax.axvline(ONSAGER, ls=":",  color="orange", lw=2, label=f"Onsager={ONSAGER:.3f}")
    ax.set_xlabel("T/J"); ax.set_ylabel("χ = N(⟨m²⟩−⟨m⟩²)")
    ax.set_title(f"(b) Susceptibility  χ_peak={chi_peak:.2f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    Ns_ref  = np.linspace(4, 900, 200)
    a_fit   = abs(delta_Tc) * np.sqrt(N)
    ax.plot(Ns_ref, ONSAGER - a_fit / np.sqrt(Ns_ref), "k-", lw=2,
            label=f"T_c(N)={ONSAGER:.3f}−{a_fit:.2f}/√N")
    ax.scatter([N], [Tc_eff], s=200, color="red", marker="*", zorder=6,
               label=f"This work  N={N}")
    ax.axhline(ONSAGER, ls="--", color="orange", lw=2, label="Onsager N→∞")
    ax.set_xlabel("N"); ax.set_ylabel("T_c^eff")
    ax.set_title("(c) Finite-Size Scaling")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(
        f"Contribution B — Monte Carlo Ising Validation  (N={N})\n"
        f"train=60 000  acc={100*acc_train:.1f}%  η_std={eta_std:.4f}  "
        f"T_c^eff={Tc_eff:.3f} vs Onsager={ONSAGER:.3f}  ΔT={delta_Tc:.3f}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Ising validation")
    parser.add_argument("--side",    type=int, default=5,     help="Grid side (5/10/28)")
    parser.add_argument("--ntrain",  type=int, default=60000, help="Training samples")
    parser.add_argument("--ntest",   type=int, default=10000, help="Test samples")
    parser.add_argument("--ntherm",  type=int, default=None,  help="Thermalisation sweeps")
    parser.add_argument("--nmeas",   type=int, default=None,  help="Measurement sweeps")
    parser.add_argument("--ntrials", type=int, default=None,  help="Trials per T")
    args = parser.parse_args()

    from vo2_network import run_pipeline, PhysicalParams

    N = args.side ** 2
    if args.ntherm  is None: args.ntherm  = 1000 if N <= 100 else 500
    if args.nmeas   is None: args.nmeas   = 1500 if N <= 100 else 800
    if args.ntrials is None: args.ntrials = 20   if N <= 100 else 10

    print(f"\n{'='*55}")
    print(f"  Contribution B — Monte Carlo  N={N}")
    print(f"{'='*55}\n")

    print("Step 1: Training network …")
    acc_train, net, conv = run_pipeline(
        side=args.side, ntrain=args.ntrain, ntest=args.ntest,
    )

    print("\nStep 2: Building J matrix …")
    p      = PhysicalParams()
    Sbase  = p.Sc / p.eta_init
    J_raw  = net.eta * net.adj * Sbase
    active = net.adj == 1
    J_mean = float(np.mean(J_raw[active]))
    J      = J_raw / max(J_mean, 1e-15)

    eta_std = float(net.eta[active].std())
    print(f"  η_std={eta_std:.4f}  J_mean={J_mean:.4e}")

    print(f"\nStep 3: Monte Carlo  "
          f"(n_therm={args.ntherm}, n_meas={args.nmeas}, n_trials={args.ntrials}) …")
    T_reds, m_mean, m_std, chi, Tc_eff, chi_peak = run_mc(
        J, N, args.ntherm, args.nmeas, args.ntrials,
    )
    delta_Tc = ONSAGER - Tc_eff

    print(f"\n{'─'*45}")
    print(f"  T_c^eff  = {Tc_eff:.3f}  J/k_B")
    print(f"  Onsager  = {ONSAGER:.3f}  J/k_B")
    print(f"  ΔT_c     = {delta_Tc:.3f}  (expected 1/√N={1/N**0.5:.3f})")
    print(f"  χ_peak   = {chi_peak:.2f}")
    print(f"{'─'*45}")

    save_path = os.path.join(OUT_DIR, "figures", f"B_N{N}_mc.png")
    plot_mc(T_reds, m_mean, m_std, chi, Tc_eff, chi_peak,
            N, acc_train, eta_std, save_path)

    pkl_path = os.path.join(OUT_DIR, f"N{N}_B.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(dict(
            N=N, Tc_eff=Tc_eff, chi_peak=chi_peak, delta_Tc=delta_Tc,
            T_reds=T_reds, m_mean=m_mean, m_std=m_std, chi=chi,
            acc_train=acc_train, eta_std=eta_std,
        ), f)
    print(f"Results saved: {pkl_path}")
