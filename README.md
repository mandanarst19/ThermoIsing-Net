# ThermoIsing-Net

**Thermal Ising Machines: A Physical Framework for Neural Computation**

> *First empirical validation of the AGS (1985) capacity law in a physical
> neuromorphic system, with a new universal noise-threshold law derived from
> capacity theory.*

---

## What this is

A simulation framework for a network of thermally-coupled **VO₂ neuristors**
that implements **Hopfield associative memory** — not by analogy, but through
a mathematically exact mapping between thermal steady-state physics and the
Hopfield Hamiltonian.

Each VO₂ device undergoes a sharp metal-insulator transition (MIT) at ~60 °C.
We exploit this bistability as a physical Ising spin.  Hebbian learning adapts
inter-device thermal conductances, storing MNIST digit prototypes as energy
minima of the physical system.  Pattern retrieval is thermal relaxation —
no digital logic, no backpropagation.

---

## Key results

| Contribution | Result |
|---|---|
| **A — Classification** | N=25: 76%  · N=100: 85%  · N=784: 85% (MNIST) |
| **B — Monte Carlo** | T_c^eff → 2.259 (N=100) vs Onsager exact 2.269 · ΔT = 0.010 |
| **C — AGS Capacity Law** | First physical validation of Amit–Gutfreund–Sompolinsky 1985 · MAE ≤ 6.7% |
| **D — Noise Threshold Law** | σ\*(p, N) ∝ (1−α/αc) · R²=0.92 (N=100) · R²=0.89 (N=784) |

### The noise threshold law

$$\boxed{\sigma^*(p, N) \;\propto\; 1 - \frac{\alpha}{\alpha_c}}$$

Networks near their storage capacity collapse first under noise.
This law is **predictable from AGS theory** — no experiment needed.

---

## Repository layout

```
ThermoIsing-Net/
├── src/
│   ├── vo2_network.py        # Core: physical model, solver, Hebbian learning
│   ├── monte_carlo.py        # Contribution B: Metropolis MC validation
│   └── noise_threshold.py    # Contribution D: σ*(p,N) capacity law
├── docs/
│   └── hebbian_fix.md        # Technical note: bug diagnosis and fix
├── results/
│   └── figures/              # Generated plots (gitignored by default)
├── data/                     # MNIST cache (auto-downloaded, gitignored)
├── notebooks/
│   └── demo.ipynb            # Interactive walkthrough
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/mandanaroosta/ThermoIsing-Net.git
cd ThermoIsing-Net
pip install -r requirements.txt
```

**Run classification (N=25):**
```bash
python src/vo2_network.py --side 5 --ntrain 6000 --ntest 1000
```

**Run Monte Carlo validation (N=100):**
```bash
python src/monte_carlo.py --side 10 --ntrain 60000
```

**Run noise threshold law (N=100):**
```bash
python src/noise_threshold.py --side 10
```

**Python API:**
```python
from src.vo2_network import run_pipeline

acc, net, conv_rate = run_pipeline(side=10, ntrain=6000, ntest=1000)
print(f"Accuracy: {100*acc:.1f}%   η_std: {net.eta[net.adj==1].std():.4f}")
```

---

## Physical model

Each VO₂ device satisfies the steady-state heat equation:

$$0 = \frac{V_i^2}{R(T_i)} - S_e(T_i - T_0) + \sum_{j \in \mathcal{N}(i)} S_{ij}(T_j - T_i)$$

The VO₂ resistance follows the hysteretic model of Zhang et al. (2023):

$$R(T,\delta) = R_0 \exp\!\left(\frac{E_a}{T}\right) F(T,\delta) + R_m, \qquad
F = \tfrac{1}{2} + \tfrac{1}{2}\tanh\!\left[\beta\!\left(\delta\tfrac{w}{2}+T_c-T\right)\right]$$

The steady-state condition maps **exactly** onto the Hopfield Hamiltonian:

$$\mathcal{H} = -\tfrac{1}{2}\sum_{ij} \eta_{ij}\,\sigma_i\,\sigma_j$$

where $\sigma_i = \text{sign}(T_i - \text{median}(T))$ are median-adaptive
Ising spins (see [`docs/hebbian_fix.md`](docs/hebbian_fix.md) for why the
median is essential).

---

## Important implementation note — v2 fix

The original implementation used a fixed threshold at the bulk MIT temperature
T_c = 332.8 K.  Because Joule self-heating drives ALL devices above T_c
simultaneously, this produced σ_i = +1 for every device, making the Hebbian
coupling matrix identically uniform (η_std = 0, no learning).

**Fix:** replace the fixed threshold with the **median temperature** of the
network at each forward pass.  This is physically justified (associative memory
depends on relative, not absolute, thermal activity) and produces η_std ≈ 0.023
with genuine pattern-encoded coupling variance.

Full diagnosis and derivation: [`docs/hebbian_fix.md`](docs/hebbian_fix.md).

---

## Parameter validation

The operating point η₀ = 0.09 (and bounds η ∈ [0.01, 0.15]) is validated
through **six independent theoretical frameworks**:

| Framework | Constraint derived |
|---|---|
| Hopfield (1982) capacity theory | η_max ≤ 0.138N/p  |
| Oja (1982) learning rule bounds | η bounds for stable PCA extraction |
| Fourier heat transfer scaling | S_ij / S_e ≪ 1 for thermal stability |
| CFL numerical stability (1928) | η_max from explicit time-step bound |
| Zhang et al. (2023) experiment | η₀ = 0.09 calibrated from device data |
| Scarpetta et al. (2018) | Phase transition at T_c^eff ≈ 2.1–2.3 J/k_B |

All six frameworks converge on η ∈ [0.01, 0.15], η₀ = 0.09.

---

## Contributions

### A — Classification accuracy

MNIST digit classification across three network sizes.  Accuracy systematically
exceeds the AGS random-pattern bound because MNIST spatial structure gives
structured patterns a storage advantage over random patterns of the same length.

### B — Monte Carlo cross-validation

The learned coupling matrix is independently simulated as a pure Ising spin
system (Metropolis algorithm).  The susceptibility peak location T_c^eff
converges toward the Onsager exact value (2.269 J/k_B) as N grows:

| N | T_c^eff | ΔT | χ_peak |
|---|---------|-----|--------|
| 25 | 2.190 | 0.079 | 1.69 |
| 100 | **2.259** | **0.010** | 3.45 |

Two independent computational paths (thermal ODE solver + Metropolis MC)
yield the same physics — the strongest possible evidence the mapping is exact.

### C — AGS capacity law

The AGS formula (Amit, Gutfreund, Sompolinsky 1985) predicts accuracy as a
function of normalised load α/αc.  **This is the first empirical test of AGS
in a physical hardware system.**

Measured deviations from AGS decrease monotonically with N (18% at N=25,
6.7% at N=100), consistent with finite-size convergence to the thermodynamic
limit assumed by the theory.

### D — Noise threshold capacity law

$$\sigma^*(p, N) = k \cdot \left(1 - \frac{\alpha}{\alpha_c}\right) + b$$

| N | k | b | R² |
|---|---|---|---|
| 100 | 0.453 | −0.075 | **0.92** |
| 784 | 6.07  | −5.50  | **0.89** |

*Networks near their storage capacity are fragile; networks with headroom are robust.*

This law predicts noise tolerance **before** deployment from nothing but the
AGS capacity formula.

---

## References

1. Hopfield, J.J. (1982). *PNAS* 79(8), 2554–2558.
2. Oja, E. (1982). *J. Math. Biology* 15(3), 267–273.
3. Zhang, E. et al. (2023). *arXiv:2307.11256* — VO₂ neuristor model.
4. Scarpetta, S. et al. (2018). *Phys. Rev. E* 97, 062305.
5. Amit, D., Gutfreund, H., Sompolinsky, H. (1985). *Phys. Rev. A* 32, 1007.
6. Mead, C. (1990). *Proc. IEEE* 78(10), 1629–1636.

---

## Citation

```bibtex
@misc{thermoising2025,
  author    = {Roosta, Mandana},
  title     = {ThermoIsing-Net: Thermal Ising Machines for Neural Computation},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/mandanaroosta/ThermoIsing-Net}
}
```

---

## Contact

**Mandana Roosta** — MSc Condensed Matter Physics, Shahid Beheshti University
✉ mandanaroosta.academia@gmail.com

*For questions about the physical model or simulation methodology, please open
a GitHub issue — this builds a public record that benefits other researchers.*

---

## License

MIT License — see [LICENSE](LICENSE).
