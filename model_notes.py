# Physical Model Notes

Physical model implementation for the thermal VO₂ network in `src/vo2_network.py`.

---

## 1. Single-device thermal model

Each VO₂ device satisfies the ordinary differential equation (Zhang et al. 2023, Eq. S6):

$$C_\text{th} \frac{dT_i}{dt} = \frac{V_i^2}{R(T_i)} - S_e(T_i - T_0) + \sum_{j \in \mathcal{N}(i)} S_{ij}(T_j - T_i)$$

**Parameters:**

| Symbol | Description | Value |
|---|---|---|
| $C_\text{th}$ | Thermal capacitance | 49.6 pJ/K |
| $S_e$ | Device-to-environment conductance | 0.201 mW/K |
| $T_0$ | Ambient temperature | 325 K |
| $T_c$ | IMT critical temperature | 332.8 K |
| $R(T)$ | Hysteretic VO₂ resistance | see Section 2 |
| $S_{ij} = \eta_{ij} S_\text{base}$ | Inter-device conductance | learnable |

**Thermal time constant:**

$$\tau = \frac{C_\text{th}}{S_e} = \frac{49.6 \times 10^{-12}}{0.201 \times 10^{-3}} \approx 228\,\text{ns}$$

**Steady-state solution**

Setting $dT_i/dt = 0$ and solving for $T_i$:

$$T_i = \frac{P_{\text{Joule},i} + S_e T_0 + \sum_j S_{ij} T_j}{S_e + \sum_j S_{ij}}$$

We solve this iteratively (Gauss–Seidel / Newton–Raphson) because $P_\text{Joule} = V_i^2 / R(T_i)$ depends nonlinearly on $T_i$.

---

## 2. VO₂ resistance model (hysteresis)

The resistance follows (Zhang et al. 2023, Eq. S7):

$$R(T, \delta) = R_0 \exp\!\left(\frac{E_a}{T}\right) F(T, \delta) + R_m$$

**Hysteresis function:**

$$F(T, \delta) = \frac{1}{2} + \frac{1}{2} \tanh\!\left[\beta\left(\delta \frac{w}{2} + T_c - T\right)\right]$$

- $\delta = +1$: device is on the **heating branch** (insulating → metallic)
- $\delta = -1$: device is on the **cooling branch** (metallic → insulating)

---

## 3. Thermal coupling

The coupling term $S_{ij} = \eta_{ij} \cdot S_\text{base}$ links thermal dynamics between nearest-neighbour devices. At the initial value $\eta_0 = 0.09$:

$$S_\text{base} = \frac{S_c}{\eta_0} = \frac{4.11 \,\mu\text{W/K}}{0.09} \approx 45.7\,\mu\text{W/K}$$

---

## 4. Input encoding

Pixel intensities $p \in [0, 1]$ are mapped to input voltages:

$$V = V_\min + (V_\max - V_\min)\sqrt{p}$$

The square-root transform improves dynamic range for dark pixels (small $p$) and ensures most inputs fall above the spiking threshold $V_\text{th} = 10.5$ V.

---

## 5. Hebbian learning rule

After each input presentation, the coupling between neighbouring devices is updated:

$$\eta_{ij} \leftarrow \mathrm{clip}\!\left(\eta_{ij} + \alpha\,\sigma_i\,\sigma_j,\;\eta_\min,\;\eta_\max\right)$$

where $\sigma_i = \mathrm{sign}(T_i - T_c) \in \{-1, +1\}$ is the Ising spin.

**Oja's rule**

Without the explicit clip operation, this is the standard Hebb rule $\Delta w_{ij} = \alpha x_i x_j$, which is unstable (weights grow without bound). The physical bounds $[\eta_\min, \eta_\max]$ implement the implicit normalisation of Oja (1982), allowing stable extraction of principal components from the input distribution.

**Hopfield networks**

The spin representation maps the VO₂ network to an Ising model. The learnable coupling matrix $\eta_{ij}$ plays the role of the synaptic weight matrix in a Hopfield network, and the steady-state thermal configuration represents an energy minimum of $E = -\frac{1}{2}\sum_{ij} \eta_{ij} \sigma_i \sigma_j$.

---

## 6. Ising–thermal mapping

The steady-state temperature profile can be mapped to an Ising spin glass by defining:

$$\sigma_i = \mathrm{sign}(T_i - T_c), \qquad \sigma_i \in \{-1, +1\}$$

The coupling $S_{ij}$ plays the role of an exchange interaction $J_{ij}$. Near the spinodal line ($T \approx T_c$), the VO₂ system exhibits scale-free avalanches and critical behaviour characteristic of a first-order transition (Scarpetta et al. 2018, Phys. Rev. E 97, 062305).

---

## References

- Zhang, E. et al. (2023). *Reconfigurable cascaded thermal neuristors for neuromorphic computing.*
- Oja, E. (1982). Simplified neuron model as a principal component analyser. *J. Math. Biology* 15(3), 267–273.
- Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS* 79(8), 2554–2558.
- Scarpetta, S. et al. (2018). Hysteresis, neural avalanches, and critical behavior near a first-order transition of a spiking neural network. *Phys. Rev. E* 97, 062305.
