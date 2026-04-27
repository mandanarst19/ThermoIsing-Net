# ThermoIsing-Net
Thermal Ising Machines: A Physical Framework for Neural Computation
## Abstract

We present a neuromorphic computing framework based on a 5 × 5 network of
thermally-coupled vanadium dioxide (VO₂) neuristors, leveraging the material's
first-order insulator-to-metal phase transition as the fundamental computational
primitive. Each device is governed by Joule self-heating and substrate-mediated
thermal coupling to its nearest neighbours. We implement an unsupervised Hebbian
learning rule that adapts inter-device thermal conductances, strengthening
connections between co-activated devices and weakening anti-correlated pairs.
The thermal dynamics act as a physical reservoir computing layer, projecting
28 × 28 MNIST digit images into a 25-dimensional thermal feature space. Combined
with PCA dimensionality reduction and a Ridge Regression readout, the system
achieves **53% accuracy** on MNIST digit classification — demonstrating that
meaningful pattern recognition can be implemented through emergent thermal
dynamics without conventional digital logic.



---

## Key Results

| Metric | Value |
|---|---|
| Network size | 5 × 5 (25 VO₂ devices) |
| Training samples | 5 000 (MNIST) |
| Test samples | 1 000 (MNIST) |
| Test accuracy | **53%** |
| Thermal convergence rate | ~95% of inputs |
| Couplings strengthened after learning | ~60% of edges |
| Thermal time constant τ | 228 ns |
| Coupling η range | [0.01, 0.15] |
| Feature dimensionality (reservoir → readout) | 784 → 50 → 25 → 10 |

---

## Repository Structure

```
vo2-neuristor-network/
│
│   └── vo2_network.py        ← Main simulation: physical model, solver,
│                                Hebbian learning, and MNIST pipeline

├── figures
├── README.md
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── LICENSE
└── .gitignore
```

---

## Physical Model

The simulation is grounded in the model of Zhang et al. (2023) for a single VO₂ neuristor device. At steady state, the heat balance for device *i* is:

$$0 = \frac{V_i^2}{R(T_i)} - S_e(T_i - T_0) + \sum_{j \in \mathcal{N}(i)} S_{ij}(T_j - T_i)$$

where:
- $V_i$ is the input voltage (mapped from pixel intensity via $V = V_\min + (V_\max - V_\min)\sqrt{p}$)
- $R(T_i)$ is the hysteretic VO₂ resistance (insulator-to-metal transition at $T_c = 332.8$ K)
- $S_e = 0.201$ mW/K is the device-to-environment thermal conductance
- $S_{ij} = \eta_{ij} \cdot S_\text{base}$ is the learnable inter-device thermal conductance

The hysteresis model follows:

$$R(T) = R_0 \exp\!\left(\frac{E_a}{T}\right) F(T, \delta) + R_m, \qquad
F = \tfrac{1}{2} + \tfrac{1}{2}\tanh\!\left[\beta\left(\delta\tfrac{w}{2} + T_c - T\right)\right]$$

where $\delta = +1$ on the heating branch and $\delta = -1$ on the cooling branch.

**Hebbian learning rule:**

$$\eta_{ij} \leftarrow \mathrm{clip}\!\left(\eta_{ij} + \alpha \,\sigma_i \sigma_j,\;\eta_\min,\;\eta_\max\right), \qquad \sigma_i = \mathrm{sign}(T_i - T_c)$$

This is equivalent to a normalised Hebbian rule (cf. Oja 1982) where the physical clipping bounds provide implicit weight normalisation.

---

### Requirements

- Python ≥ 3.9
- NumPy, scikit-learn, matplotlib (see `requirements.txt` for pinned versions)
- Internet connection required on first run to download MNIST (~11 MB via OpenML)

---

## Quick Start

```python
from src.vo2_network import MNISTClassifier, plot_results

# Instantiate the full pipeline
clf = MNISTClassifier()

# Load MNIST (downloads automatically on first run)
(Xtrain, ytrain), (Xtest, ytest) = clf.load_data(ntrain=5000, ntest=1000)

# Train: Hebbian unsupervised pass + PCA + Ridge readout
train_acc = clf.train(Xtrain, ytrain)

# Evaluate on held-out test data
test_acc, y_pred = clf.test(Xtest, ytest)

# Save diagnostic figures
plot_results(clf, savedir="results/figures/")

print(f"Test accuracy: {100*test_acc:.1f}%")
```

**Expected console output:**

```
============================================================
  Thermal VO₂ Neuristor Network — MNIST Classification
  2025-XX-XX  XX:XX:XX
============================================================
Loading MNIST …
  Loaded 5000 train + 1000 test samples.

Training  (5000 samples) …
============================================================
  [  100/5000]  η ∈ [0.010, 0.150]  conv: 94.0%
  ...
  Convergence rate: 4750/5000 (95.0%)

  Learned couplings:
    Strengthened : 24/40  (60.0%)
    Weakened     : 16/40  (40.0%)
    η range      : [0.010, 0.150]

  PCA: 50D → 25D …
  Training Ridge readout …
  Train accuracy: 58.3%

Testing  (1000 samples) …
  Convergence rate: 952/1000 (95.2%)
  Test accuracy: 53.0%

============================================================
  RESULTS SUMMARY
============================================================
  Train accuracy         : 58.3%
  Test  accuracy         : 53.0%
  Couplings strengthened : 60.0%
  Couplings weakened     : 40.0%
============================================================
```

---

## Usage

### Run from the command line

```bash
python src/vo2_network.py
```

Results, figures, and the serialised model will be written to `results/`.

### Explore individual components

```python
from src.vo2_network import (
    PhysicalParams, ThermalSolver, HebbianNetwork, make_grid_5x5
)
import numpy as np

# Inspect physical parameters
p = PhysicalParams()
print(f"Transition temperature: {p.Tc} K")
print(f"Thermal time constant τ = Cth/Se = {p.Cth/p.Se*1e9:.0f} ns")

# Solve a single thermal state
solver = ThermalSolver(n=25)
adj    = make_grid_5x5()
net    = HebbianNetwork()
V      = np.full(25, 14.0)          # uniform 14 V input
T, ok  = solver.solve(V, net.eta, adj)
print(f"Converged: {ok}, max T = {T.max():.2f} K")
```

---

## Reproducing the Paper Results

The results in the manuscript are obtained with the default hyperparameters in `PhysicalParams` and the following call:

```python
clf.load_data(ntrain=5000, ntest=1000)
clf.train(Xtrain, ytrain)
clf.test(Xtest, ytest)
```

Note on stochasticity: the PCA and Ridge steps involve no randomness given fixed training features. Slight run-to-run variation in accuracy (±1–2%) can arise from floating-point non-determinism in the iterative thermal solver on different hardware.

---

## References

1. Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS* 79(8), 2554–2558.
2. Oja, E. (1982). Simplified neuron model as a principal component analyser. *J. Math. Biology* 15(3), 267–273.
3. Zhang, E. et al. (2023). Reconfigurable cascaded thermal neuristors for neuromorphic computing. *arXiv:2307.11256*.
4. Scarpetta, S., Apicella, I., Minati, L., & De Candia, A. (2018). Hysteresis, neural avalanches, and critical behavior near a first-order transition of a spiking neural network. *Phys. Rev. E* 97, 062305.
5. Mead, C. (1990). Neuromorphic electronic systems. *Proc. IEEE* 78(10), 1629–1636.

---

---

## License

This project is licensed under the MIT License. See [LICENSE] for details.



## Contact

Mandana Roosta  
PhD candidate  
Physics Department,Shahid Beheshti University  
✉ mandanaroosta.academia@gmail.com 


*For questions about the physical model or simulation methodology, please open a GitHub issue rather than sending email — this helps build a public record that benefits other researchers.*
