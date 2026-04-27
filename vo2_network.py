"""
vo2_network.py
==============
Thermal VO₂ Neuristor Network for Neuromorphic Computing

Implements a 5×5 network of thermally-coupled vanadium dioxide (VO₂) neuristors
that perform unsupervised feature extraction via Hebbian learning, followed by
PCA dimensionality reduction and Ridge Regression classification on the MNIST
handwritten digit dataset.

Architecture overview
---------------------
    MNIST image (28×28)
        │
        ▼  block-average downsample
    5×5 pixel grid
        │
        ▼  square-root voltage mapping  V ∈ [8, 24] V
    25 input voltages
        │
        ▼  steady-state thermal solver (Newton–Raphson iteration)
    25 device temperatures  T_i
        │
        ▼  feature extraction  (binary spins + normalised temperatures)
    50-dimensional feature vector
        │
        ▼  PCA  50D → 25D
    principal components
        │
        ▼  Ridge Regression
    digit class (0–9)

Physical model
--------------
Each VO₂ device satisfies the steady-state heat equation

    0 = P_Joule(i) − S_e (T_i − T_0) + Σ_j S_ij (T_j − T_i)

where P_Joule = V_i² / R(T_i) is Joule heating, S_e is the device-to-environment
thermal conductance, S_ij = η_ij · S_base is the learnable inter-device coupling,
and R(T) is the hysteretic VO₂ resistance from Zhang et al. (2023), Eq. (S7).

The coupling strengths η_ij are updated by a normalised Hebbian rule

    Δη_ij = α · σ_i · σ_j,   σ_i = sign(T_i − T_c)

clipped to [η_min, η_max] for stability (analogous to Oja's rule, 1982).

References
----------
[1]  Hopfield, J.J. (1982). Neural networks and physical systems with emergent
     collective computational abilities. PNAS 79(8), 2554–2558.
[2]  Oja, E. (1982). Simplified neuron model as a principal component analyser.
     J. Math. Biology 15(3), 267–273.
[3]  Zhang, E. et al. (2023). Reconfigurable cascaded thermal neuristors for
     neuromorphic computing. arXiv:2307.11256.
[4]  Scarpetta, S. et al. (2018). Hysteresis, neural avalanches, and critical
     behavior near a first-order transition of a spiking neural network.
     Phys. Rev. E 97, 062305.

Author
------
[Mandana Roosta : Your Name]
[Shahid Beheshti University]
[mandanaroosta.academia@gmail.com]
"""

from __future__ import annotations

import os
import pickle
import warnings
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.datasets import fetch_openml

warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings


# =============================================================================
# Physical parameters (Zhang et al. 2023, Table I & Supplementary)
# =============================================================================

class PhysicalParams:
    """
    Material and circuit parameters for a single VO₂ neuristor device.

    All values are taken from Zhang et al. (2023), Table I and the
    Supplementary Material, unless noted otherwise.

    Attributes
    ----------
    Cth : float
        Thermal capacitance of the device [J K⁻¹].
    Se : float
        Thermal conductance to the environment [W K⁻¹].
    Sc : float
        Reference inter-device thermal conductance [W K⁻¹].
    T0 : float
        Ambient (substrate) temperature [K].
    Tc : float
        VO₂ insulator-to-metal transition temperature [K] (Morin, 1959).
    C : float
        Parasitic capacitance of the VO₂ nanodevice [F].
    Rload : float
        Load resistance in series with the device [Ω].
    R0 : float
        Pre-exponential resistance factor for the insulating state [Ω].
    Ea : float
        Activation energy of the insulating state expressed as a
        temperature scale [K].  Ea_eV ≈ 0.45 eV.
    Rm : float
        Metallic-state (channel) resistance [Ω].
    w : float
        Hysteresis half-width in the tanh model [K].
    beta : float
        Sharpness parameter of the hysteresis transition [K⁻¹].
    gamma : float
        Fitted scaling factor for metallic-channel formation.
    Vmin, Vmax : float
        Input voltage range mapped to pixel intensities [0, 1] [V].
    Vth : float
        Approximate threshold voltage for spiking onset [V].
    alpha : float
        Hebbian learning rate (dimensionless).
    eta_min, eta_max : float
        Bounds on the learnable coupling strength η (dimensionless).
    eta_init : float
        Initial uniform coupling strength η₀.
    """

    # --- Thermal ---
    Cth: float = 49.6e-12   # J/K
    Se: float  = 0.201e-3   # W/K  (environment conductance)
    Sc: float  = 4.11e-6    # W/K  (inter-device conductance at η = 1)
    T0: float  = 325.0      # K    (substrate temperature)
    Tc: float  = 332.8      # K    (IMT critical temperature)

    # --- Electrical ---
    C: float     = 145e-12  # F
    Rload: float = 12.0e3   # Ω

    # --- VO₂ resistance hysteresis model (Zhang Eq. S7) ---
    R0: float    = 5.36e-3  # Ω   (pre-exponential)
    Ea: float    = 5220.0   # K   (activation energy scale)
    Rm: float    = 1286.0   # Ω   (metallic-state resistance)
    w: float     = 7.19     # K   (hysteresis half-width)
    beta: float  = 0.253    # K⁻¹ (transition sharpness)
    gamma: float = 0.956    #     (metallic-channel scaling)

    # --- Voltage mapping ---
    Vmin: float = 8.0       # V   (voltage at pixel intensity 0)
    Vmax: float = 24.0      # V   (voltage at pixel intensity 1)
    Vth: float  = 10.5      # V   (approximate spiking threshold)

    # --- Hebbian learning ---
    alpha: float    = 0.001 #     (learning rate)
    eta_min: float  = 0.01  #     (minimum coupling; prevents decoupling)
    eta_max: float  = 0.15  #     (maximum coupling; stability bound)
    eta_init: float = 0.09  #     (uniform initial coupling)


# =============================================================================
# Topology
# =============================================================================

def make_grid_5x5() -> np.ndarray:
    """
    Build the adjacency matrix for a 5×5 four-connected (von Neumann) grid.

    Node numbering follows row-major order::

        0  1  2  3  4
        5  6  7  8  9
       10 11 12 13 14
       15 16 17 18 19
       20 21 22 23 24

    Each node is connected to its horizontal and vertical neighbours only
    (no diagonal connections), reflecting the physical layout where nearest-
    neighbour devices share a thermal conduction path through the substrate.

    Returns
    -------
    adj : np.ndarray, shape (25, 25), dtype float64
        Symmetric binary adjacency matrix; ``adj[i, j] = 1`` iff devices
        *i* and *j* are nearest neighbours.
    """
    n = 25
    adj = np.zeros((n, n))

    for i in range(n):
        row, col = divmod(i, 5)
        if col < 4:          # right neighbour
            adj[i, i + 1] = adj[i + 1, i] = 1
        if row < 4:          # lower neighbour
            adj[i, i + 5] = adj[i + 5, i] = 1

    return adj


# =============================================================================
# Image pre-processing
# =============================================================================

def downsample_image(img28: np.ndarray) -> np.ndarray:
    """
    Downsample a 28×28 greyscale image to a 5×5 representation by block
    averaging.

    Each of the 25 output pixels corresponds to a non-overlapping rectangular
    patch of the original image. The patch boundaries are determined by
    uniformly tiling ``floor(k × 28/5) : floor((k+1) × 28/5)`` for
    k ∈ {0, …, 4} along each axis, yielding patches of approximately
    5 or 6 pixels per side.

    Parameters
    ----------
    img28 : np.ndarray, shape (28, 28)
        Normalised greyscale image with pixel values in [0, 1].

    Returns
    -------
    img5 : np.ndarray, shape (5, 5)
        Block-averaged downsampled image, values in [0, 1].
    """
    img5 = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            r1, r2 = int(i * 28 / 5), int((i + 1) * 28 / 5)
            c1, c2 = int(j * 28 / 5), int((j + 1) * 28 / 5)
            img5[i, j] = np.mean(img28[r1:r2, c1:c2])
    return img5


# =============================================================================
# Thermal solver
# =============================================================================

class ThermalSolver:
    """
    Iterative steady-state solver for the coupled VO₂ thermal network.

    At steady state, the heat equation for device *i* reduces to

    .. math::

        T_i = \\frac{P_{\\text{Joule},i} + S_e T_0 + \\sum_j S_{ij} T_j}
                    {S_e + \\sum_j S_{ij}}

    Because the Joule power P_Joule = V_i² / R(T_i) depends non-linearly
    on temperature through the hysteretic resistance R(T), this equation
    is solved by Gauss–Seidel iteration (analogous to Newton–Raphson with
    Patankar under-relaxation) until convergence in the L∞ norm.

    Hysteresis state (heating vs. cooling) is tracked across calls to
    ``solve`` so that the device correctly follows its hysteresis branch
    during sequential inference or training.

    Parameters
    ----------
    n : int
        Number of devices in the network (default 25 for a 5×5 grid).
    """

    def __init__(self, n: int = 25) -> None:
        self.n: int = n
        self.p: PhysicalParams = PhysicalParams()
        self.heating: np.ndarray = np.zeros(n, dtype=int)   # +1 = heating branch
        self.Tprev: np.ndarray  = np.ones(n) * self.p.T0

    def get_resistance(
        self,
        T: np.ndarray,
        Tprev: np.ndarray,
        state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the hysteretic VO₂ resistance at each device.

        The model follows Zhang et al. (2023), Eq. (S7):

        .. math::

            R(T) = R_0 \\exp\\!\\left(\\frac{E_a}{T}\\right) F(T, \\delta)
                   + R_m

        where the hysteresis function is

        .. math::

            F(T, \\delta) = \\tfrac{1}{2}
                + \\tfrac{1}{2} \\tanh\\!\\left[
                    \\beta \\left(\\delta \\tfrac{w}{2} + T_c - T\\right)
                \\right]

        with δ = +1 on the heating branch and δ = −1 on the cooling branch.

        Parameters
        ----------
        T : np.ndarray, shape (n,)
            Current device temperatures [K].
        Tprev : np.ndarray, shape (n,)
            Device temperatures from the previous iteration [K].
        state : np.ndarray, shape (n,)
            Previous hysteresis branch indicators (±1).

        Returns
        -------
        R : np.ndarray, shape (n,)
            Device resistances [Ω].
        delta : np.ndarray, shape (n,)
            Updated branch indicators (±1).
        """
        delta = np.sign(T - Tprev)
        delta[delta == 0] = state[delta == 0]   # keep branch when ΔT = 0

        Rins = self.p.R0 * np.exp(self.p.Ea / T)
        arg  = self.p.beta * (delta * self.p.w / 2 + self.p.Tc - T)
        F    = 0.5 + 0.5 * np.tanh(arg)
        R    = Rins * F + self.p.Rm

        return R, delta

    def solve(
        self,
        V: np.ndarray,
        eta: np.ndarray,
        adj: np.ndarray,
        maxiter: int = 500,
        tol: float = 1e-3,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve for the steady-state temperature vector of the network.

        Parameters
        ----------
        V : np.ndarray, shape (n,)
            Applied input voltages [V].
        eta : np.ndarray, shape (n, n)
            Current thermal coupling matrix (learnable weights, η_ij ≥ 0).
        adj : np.ndarray, shape (n, n)
            Binary adjacency matrix encoding the network topology.
        maxiter : int
            Maximum number of Gauss–Seidel iterations (default 500).
        tol : float
            Convergence tolerance in the L∞ norm [K] (default 1e-3 K).

        Returns
        -------
        T : np.ndarray, shape (n,)
            Steady-state device temperatures [K].
        converged : bool
            ``True`` if the iteration converged within ``maxiter`` steps.
        """
        T = np.ones(self.n) * self.p.T0
        R = np.ones(self.n) * 1e4          # start in insulating state

        # Convert dimensionless η to physical conductance [W K⁻¹]
        # S_base ≈ 46 μW K⁻¹ ensures S_ij(η_init) = Sc
        S_base = self.p.Sc / self.p.eta_init
        S = eta * adj * S_base             # shape (n, n)

        for _ in range(maxiter):
            Told = T.copy()
            R, self.heating = self.get_resistance(T, self.Tprev, self.heating)

            for i in range(self.n):
                P_joule      = V[i] ** 2 / R[i]
                neighbor_sum = np.sum(S[i, :] * T)
                G_total      = self.p.Se + np.sum(S[i, :])
                T[i]         = (P_joule + self.p.Se * self.p.T0 + neighbor_sum) / G_total

            if np.max(np.abs(T - Told)) < tol:
                self.Tprev = T.copy()
                return T, True

        return T, False     # did not converge


# =============================================================================
# Hebbian network
# =============================================================================

class HebbianNetwork:
    """
    5×5 thermal VO₂ network with unsupervised Hebbian learning.

    The network implements a physical analogue of Hebbian/Oja learning.
    After each input presentation, coupling strengths are updated as

    .. math::

        \\eta_{ij} \\leftarrow
        \\mathrm{clip}\\!\\left(
            \\eta_{ij} + \\alpha \\, \\sigma_i \\sigma_j,\\;
            \\eta_{\\min},\\; \\eta_{\\max}
        \\right)

    where σ_i = sign(T_i − T_c) ∈ {−1, +1} is the Ising spin of device *i*.

    Because the physical system naturally bounds η via the clip operation
    (encoding both a biological saturation mechanism and thermal stability
    constraints), the learning rule is stable without explicit weight-decay
    terms, reproducing the key property of Oja's normalised Hebbian rule.

    Attributes
    ----------
    adj : np.ndarray, shape (25, 25)
        Fixed binary adjacency matrix of the 5×5 grid.
    eta : np.ndarray, shape (25, 25)
        Learnable thermal coupling strengths (initialised uniformly).
    solver : ThermalSolver
        Handles steady-state thermal equations for each input.
    """

    def __init__(self) -> None:
        self.params: PhysicalParams = PhysicalParams()
        self.n: int = 25

        self.adj: np.ndarray = make_grid_5x5()
        self.eta: np.ndarray = self.params.eta_init * self.adj.copy()

        self.solver: ThermalSolver = ThermalSolver(n=25)
        self.eta_history: list = []

    def get_spins(self, T: np.ndarray) -> np.ndarray:
        """
        Convert device temperatures to Ising spins.

        Parameters
        ----------
        T : np.ndarray, shape (n,)
            Steady-state temperatures [K].

        Returns
        -------
        spins : np.ndarray, shape (n,), dtype int
            +1 if T_i > T_c (metallic), −1 if T_i < T_c (insulating).
        """
        return np.sign(T - self.params.Tc).astype(int)

    def get_features(self, T: np.ndarray) -> np.ndarray:
        """
        Extract a 50-dimensional feature vector from the thermal state.

        The feature vector concatenates:
        - **Spin features** (dims 0–24): binary Ising spins σ_i ∈ {−1, +1}
          encoding whether each device is in its metallic or insulating phase.
        - **Temperature features** (dims 25–49): device temperatures
          min-max normalised to [0, 1], providing a graded representation of
          the thermal excitation level.

        Parameters
        ----------
        T : np.ndarray, shape (25,)
            Steady-state device temperatures [K].

        Returns
        -------
        features : np.ndarray, shape (50,)
            Concatenated spin and normalised temperature features.
        """
        s    = self.get_spins(T)
        Tnorm = (T - T.min()) / (T.max() - T.min() + 1e-10)
        return np.concatenate([s, Tnorm])

    def hebbian_update(self, T: np.ndarray) -> None:
        """
        Apply one step of the bounded Hebbian learning rule.

        Only neighbouring device pairs (where ``adj[i,j] = 1``) are updated,
        reflecting the physical locality of thermal coupling through the
        substrate. Co-activated pairs (σ_i σ_j > 0) are strengthened;
        anti-correlated pairs are weakened.

        Parameters
        ----------
        T : np.ndarray, shape (25,)
            Steady-state temperatures from the current forward pass [K].
        """
        s = self.get_spins(T)
        for i in range(self.n):
            for j in range(self.n):
                if self.adj[i, j] == 1:
                    delta = self.params.alpha * s[i] * s[j]
                    self.eta[i, j] = np.clip(
                        self.eta[i, j] + delta,
                        self.params.eta_min,
                        self.params.eta_max,
                    )

    def process_image(
        self,
        img: np.ndarray,
        learn: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Run the full forward pass for a single 28×28 MNIST image.

        Pipeline:
        1. Downsample 28×28 → 5×5 by block averaging.
        2. Apply square-root voltage mapping: V = V_min + (V_max − V_min)√p,
           where p ∈ [0,1] is the pixel intensity.  The √ transform spreads
           dark pixels (small p) more evenly across the voltage range, reducing
           the fraction of sub-threshold inputs.
        3. Solve the coupled thermal equations for steady-state temperatures.
        4. Extract the 50-dimensional feature vector.
        5. (Training only) Apply the Hebbian coupling update.

        Parameters
        ----------
        img : np.ndarray, shape (28, 28)
            Normalised greyscale pixel array, values in [0, 1].
        learn : bool
            If ``True``, update coupling strengths after the forward pass.

        Returns
        -------
        features : np.ndarray, shape (50,)
            Extracted feature vector.  Zero vector if the solver diverged.
        converged : bool
            ``True`` if the thermal solver converged for this image.
        """
        img5 = downsample_image(img)

        # Square-root voltage mapping (expands dynamic range for dark pixels)
        V = self.params.Vmin + (self.params.Vmax - self.params.Vmin) * np.sqrt(img5)
        V = V.flatten()

        T, ok = self.solver.solve(V, self.eta, self.adj)

        if not ok:
            return np.zeros(50), False

        feats = self.get_features(T)

        if learn:
            self.hebbian_update(T)

        return feats, True


# =============================================================================
# Full classification pipeline
# =============================================================================

class MNISTClassifier:
    """
    End-to-end MNIST classification pipeline using the thermal VO₂ network.

    Stages
    ------
    1. **Reservoir** (``HebbianNetwork``): unsupervised physical feature
       extraction with adaptive Hebbian coupling.
    2. **Dimensionality reduction** (``sklearn.decomposition.PCA``):
       50D thermal features → 25D principal components.
    3. **Readout** (``sklearn.linear_model.Ridge``): linear classifier
       trained on labelled PCA embeddings.

    This architecture is a form of *reservoir computing*: the physical
    nonlinear dynamics of the VO₂ network act as a fixed (but adaptive)
    reservoir, while only the readout layer is trained in the supervised sense.

    Attributes
    ----------
    net : HebbianNetwork
        The physical reservoir / feature extractor.
    pca : PCA or None
        Fitted PCA object (``None`` before training).
    clf : Ridge or None
        Fitted Ridge classifier (``None`` before training).
    stats : dict
        Training statistics (convergence rate, coupling distribution).
    """

    def __init__(self) -> None:
        self.net:   HebbianNetwork = HebbianNetwork()
        self.pca:   Optional[PCA]   = None
        self.clf:   Optional[Ridge]  = None
        self.stats: dict             = {}

    def load_data(
        self,
        ntrain: int = 5000,
        ntest:  int = 1000,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Download and split the MNIST dataset.

        Uses ``sklearn.datasets.fetch_openml`` to download MNIST-784 from
        OpenML.  Images are normalised to [0, 1] and reshaped to (28, 28).

        Parameters
        ----------
        ntrain : int
            Number of training samples (default 5 000).
        ntest : int
            Number of test samples (default 1 000).

        Returns
        -------
        (Xtrain, ytrain) : tuple
            Training images, shape (ntrain, 28, 28), and integer labels.
        (Xtest, ytest) : tuple
            Test images, shape (ntest, 28, 28), and integer labels.
        """
        print("Loading MNIST …")
        mnist = fetch_openml("mnist_784", version=1, parser="auto")
        X = mnist.data.values / 255.0
        y = mnist.target.values.astype(int)
        X = X.reshape(-1, 28, 28)
        print(f"  Loaded {ntrain} train + {ntest} test samples.")
        return (X[:ntrain], y[:ntrain]), (X[ntrain:ntrain + ntest], y[ntrain:ntrain + ntest])

    def train(
        self,
        Xtrain: np.ndarray,
        ytrain: np.ndarray,
        verbose: bool = True,
    ) -> float:
        """
        Train the network: Hebbian unsupervised pass + supervised readout.

        Parameters
        ----------
        Xtrain : np.ndarray, shape (N, 28, 28)
            Training images normalised to [0, 1].
        ytrain : np.ndarray, shape (N,)
            Integer digit labels 0–9.
        verbose : bool
            If ``True``, print progress every 100 samples.

        Returns
        -------
        train_acc : float
            Training accuracy on the labelled set (fraction in [0, 1]).
        """
        n = len(Xtrain)
        feats = np.zeros((n, 50))
        n_conv = 0

        if verbose:
            print(f"\nTraining  ({n} samples) …")
            print("=" * 60)

        for i in range(n):
            f, ok = self.net.process_image(Xtrain[i], learn=True)
            feats[i] = f
            if ok:
                n_conv += 1

            if verbose and (i + 1) % 100 == 0:
                active_eta = self.net.eta[self.net.adj == 1]
                print(
                    f"  [{i+1:>5}/{n}]  η ∈ [{active_eta.min():.3f}, "
                    f"{active_eta.max():.3f}]  conv: {100*n_conv/(i+1):.1f}%"
                )

        if verbose:
            print("=" * 60)
            print(f"  Convergence rate: {n_conv}/{n} ({100*n_conv/n:.1f}%)")

        # Analyse learned coupling distribution
        active_eta = self.net.eta[self.net.adj == 1]
        n_up   = np.sum(active_eta > self.net.params.eta_init)
        n_down = np.sum(active_eta < self.net.params.eta_init)
        n_tot  = len(active_eta)

        if verbose:
            print(f"\n  Learned couplings:")
            print(f"    Strengthened : {n_up}/{n_tot}  ({100*n_up/n_tot:.1f}%)")
            print(f"    Weakened     : {n_down}/{n_tot}  ({100*n_down/n_tot:.1f}%)")
            print(f"    η range      : [{active_eta.min():.3f}, {active_eta.max():.3f}]")

        self.stats = {
            "conv_rate": n_conv / n,
            "pct_strengthened": 100 * n_up   / n_tot,
            "pct_weakened":     100 * n_down  / n_tot,
        }

        # PCA: 50D → 25D
        if verbose:
            print("\n  PCA: 50D → 25D …")
        self.pca = PCA(n_components=25)
        f_pca = self.pca.fit_transform(feats)

        # Ridge readout (one-hot targets)
        if verbose:
            print("  Training Ridge readout …")
        Y_oh = np.zeros((n, 10))
        Y_oh[np.arange(n), ytrain] = 1.0

        self.clf = Ridge(alpha=1.0)
        self.clf.fit(f_pca, Y_oh)

        y_pred    = np.argmax(self.clf.predict(f_pca), axis=1)
        train_acc = np.mean(y_pred == ytrain)

        if verbose:
            print(f"  Train accuracy: {100*train_acc:.1f}%")

        return train_acc

    def test(
        self,
        Xtest:  np.ndarray,
        ytest:  np.ndarray,
        verbose: bool = True,
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate the trained model on held-out test data (no weight updates).

        Parameters
        ----------
        Xtest : np.ndarray, shape (M, 28, 28)
            Test images normalised to [0, 1].
        ytest : np.ndarray, shape (M,)
            Ground-truth integer labels.
        verbose : bool
            If ``True``, print convergence rate and final accuracy.

        Returns
        -------
        test_acc : float
            Classification accuracy on the test set (fraction in [0, 1]).
        y_pred : np.ndarray, shape (M,)
            Predicted digit classes.
        """
        n = len(Xtest)
        feats  = np.zeros((n, 50))
        n_conv = 0

        if verbose:
            print(f"\nTesting  ({n} samples) …")

        for i in range(n):
            f, ok = self.net.process_image(Xtest[i], learn=False)
            feats[i] = f
            if ok:
                n_conv += 1

        if verbose:
            print(f"  Convergence rate: {n_conv}/{n} ({100*n_conv/n:.1f}%)")

        f_pca  = self.pca.transform(feats)
        y_pred = np.argmax(self.clf.predict(f_pca), axis=1)
        acc    = np.mean(y_pred == ytest)

        if verbose:
            print(f"  Test accuracy: {100*acc:.1f}%")

        return acc, y_pred


# =============================================================================
# Visualisation utilities
# =============================================================================

def plot_results(classifier: MNISTClassifier, savedir: str = "./results/figures/") -> None:
    """
    Generate and save diagnostic figures for the trained network.

    Produces two figures:

    1. **coupling_evolution.png** — three-panel heatmap showing the initial
       uniform coupling matrix, the matrix after Hebbian learning, and the
       signed difference (ΔΗ = η_final − η_init).

    2. **coupling_distribution.png** — histogram of learned coupling values
       with a vertical reference line at η_init.

    Parameters
    ----------
    classifier : MNISTClassifier
        A trained classifier instance.
    savedir : str
        Directory in which to save the PNG files (created if absent).
    """
    os.makedirs(savedir, exist_ok=True)

    eta_init_mat = classifier.net.params.eta_init * classifier.net.adj
    eta_final    = classifier.net.eta

    # --- Figure 1: coupling evolution ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Thermal Coupling Evolution (Hebbian Learning)", fontsize=13)

    titles = ["Initial (uniform)", "After learning", "Δη = final − initial"]
    data   = [eta_init_mat, eta_final, eta_final - eta_init_mat]
    cmaps  = ["YlOrRd", "YlOrRd", "RdBu_r"]
    vlims  = [(0, 0.15), (0, 0.15), (-0.1, 0.1)]

    for ax, d, t, cm, (vmin, vmax) in zip(axes, data, titles, cmaps, vlims):
        im = ax.imshow(d, cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("Device column index")
        ax.set_ylabel("Device row index")
        plt.colorbar(im, ax=ax, label="η")

    plt.tight_layout()
    path1 = os.path.join(savedir, "coupling_evolution.png")
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    print(f"Saved: {path1}")
    plt.close()

    # --- Figure 2: coupling distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    active_eta = classifier.net.eta[classifier.net.adj == 1]
    ax.hist(active_eta, bins=30, color="steelblue", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(
        classifier.net.params.eta_init,
        color="crimson", linestyle="--", linewidth=1.5,
        label=f"η₀ = {classifier.net.params.eta_init}",
    )
    ax.set_xlabel("Thermal coupling strength η", fontsize=12)
    ax.set_ylabel("Number of edges", fontsize=12)
    ax.set_title("Distribution of Learned Coupling Strengths", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(savedir, "coupling_distribution.png")
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    print(f"Saved: {path2}")
    plt.close()


# =============================================================================
# Entry point
# =============================================================================

def main() -> MNISTClassifier:
    """
    Run the full training and evaluation pipeline.

    Loads MNIST, trains the thermal VO₂ network with Hebbian learning,
    evaluates on the held-out test set, generates diagnostic figures,
    and saves the trained model to ``results/model.pkl``.

    Returns
    -------
    clf : MNISTClassifier
        The fully trained classifier instance.
    """
    print("=" * 60)
    print("  Thermal VO₂ Neuristor Network — MNIST Classification")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)

    clf = MNISTClassifier()
    (Xtrain, ytrain), (Xtest, ytest) = clf.load_data(ntrain=5000, ntest=1000)

    train_acc = clf.train(Xtrain, ytrain)
    test_acc, _y_pred = clf.test(Xtest, ytest)

    print("\nGenerating figures …")
    plot_results(clf)

    # Persist model for downstream analysis
    model_path = "results/model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(
            {
                "network":  clf.net,
                "pca":      clf.pca,
                "clf":      clf.clf,
                "stats":    clf.stats,
                "test_acc": test_acc,
            },
            fh,
        )
    print(f"\nModel saved → {model_path}")

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Train accuracy         : {100*train_acc:.1f}%")
    print(f"  Test  accuracy         : {100*test_acc:.1f}%")
    print(f"  Couplings strengthened : {clf.stats['pct_strengthened']:.1f}%")
    print(f"  Couplings weakened     : {clf.stats['pct_weakened']:.1f}%")
    print("=" * 60)

    return clf


if __name__ == "__main__":
    clf = main()
