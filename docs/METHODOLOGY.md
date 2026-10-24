# FLaaS Methodology & Mathematical Foundations

## 1. Sign Convention for Client Updates & Server Optimizers

To avoid sign discrepancies across client algorithms, attacks, aggregators, and server optimizers, FLaaS standardizes on the **model delta convention**:

### Client Local Update Definition
For a client $i \in \{1, \dots, K\}$ starting from global model parameters $w^t$:
$$\Delta w_i^t \triangleq w_i^{t, E} - w^t$$
where $w_i^{t, E}$ denotes the local parameters after $E$ local epochs.

- Under standard gradient descent with learning rate $\eta_l$ and local pseudo-gradient $g_i$, $\Delta w_i^t \approx -\eta_l \tau_i g_i$.
- Therefore, $\Delta w_i^t$ represents the **direction of parameter change** (not the negative gradient).

### Server Optimization Update
The server aggregates client deltas into $\Delta^t$:
$$\Delta^t = \text{Aggregate}\left(\{\Delta w_i^t\}_{i=1}^K, \{p_i\}_{i=1}^K\right)$$

For **ServerSGD** with server learning rate $\eta_g$:
$$w^{t+1} = w^t + \eta_g \Delta^t$$
When $\eta_g = 1.0$ and sample weighting $p_i = \frac{n_i}{\sum_j n_j}$ is used, this recovers exact standard Federated Averaging (McMahan et al., 2017):
$$w^{t+1} = \sum_{i=1}^K p_i w_i^{t, E}$$

For **ServerAdam** (Reddi et al., ICLR 2021):
- First moment: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) \Delta^t$
- Second moment: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\Delta^t)^2$
- Bias correction: $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
- Parameter step: $w^{t+1} = w^t + \eta_g \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

### Adversarial Attack Representation
Under the delta sign convention:
- **Sign-Flip Attack**: $\Delta w_{\text{adv}} = -\Delta w_{\text{honest}}$ (or $-s \cdot \Delta w_{\text{honest}}$ with scaling factor $s$).
- **Gaussian Noise Attack**: $\Delta w_{\text{adv}} \sim \mathcal{N}(0, \sigma_{\text{adv}}^2 I)$.
- **Scaled Update Attack**: $\Delta w_{\text{adv}} = \gamma \cdot \Delta w_{\text{honest}}$ where $\gamma \gg 1$.
- **Label-Flip Attack**: Carried out during local dataset generation/dataloader sampling by mapping source class $y \mapsto (C - 1 - y) \pmod C$.

---

## 2. Numerical Equivalence Definitions

In floating-point arithmetic on deep learning accelerators (GPUs/CPUs), operations involving zero additions or identity scalings are subjected to standard IEEE 754 precision constraints:
- All boundary equivalences (e.g. FedProx $\mu=0$, SCAFFOLD initial $c=0$, FedNova equal steps $\tau$, TrimmedMean $\beta=0$, MultiKrum $m=n$) are evaluated as **numerically equal** under an absolute tolerance threshold:
$$\max_j |a_j - b_j| \le 10^{-6}$$
- Statements of equivalence denote numerical equivalence within `atol=1e-6`, not bit-for-bit identity across disparate compute pathways.

---

## 3. Differential Privacy: Client-Level DP vs. Example-Level DP-SGD

### 3.1 Formal Scope: Client-Level (User-Level) DP in Federated Learning
Federated Learning systems in cross-silo and cross-device environments (e.g. healthcare hospital cohorts, banking institutions, consumer edge devices) require protection against the reconstruction or inference of an entire client's dataset (McMahan et al., 2018; Bhowmick et al., 2018).

- **Example-Level DP-SGD** (Abadi et al., 2016):
  - Record universe $\mathcal{D}$: Individual data records (e.g. single clinical scans or user transactions).
  - Mechanism: Per-sample gradient clipping and noise addition during each client's local mini-batch SGD.
  - Subsampling: $q_{\text{sample}} = B / N_i$ where $B$ is batch size and $N_i$ is client dataset size.
- **Client-Level (User-Level) DP** (FLaaS Target Formulation):
  - Record universe $\mathcal{U}$: The $K$ distinct client organizations/devices in the federation.
  - Goal: Ensure that the inclusion or exclusion of any single client's entire dataset $D_i$ does not distinguishably alter the distribution of the global model parameters $w^T$.
  - Subsampling: In each communication round $t \in \{1, \dots, T\}$, a cohort $S_t \subseteq \{1, \dots, K\}$ of size $m$ is sampled uniformly at random. The client-level sampling ratio is:
    $$q = \frac{m}{K}$$

### 3.2 Sensitivity and Noise Calibration
1. **Client-Level L2 Norm Clipping**:
   Each participating client $i \in S_t$ clips its total local update $\Delta w_i^t$ to an $L_2$ threshold $C$:
   $$\Delta \tilde{w}_i^t = \Delta w_i^t \cdot \min\left(1, \frac{C}{\|\Delta w_i^t\|_2}\right)$$
   guaranteeing $\|\Delta \tilde{w}_i^t\|_2 \le C$.

2. **Global Sensitivity of the Sum**:
   Under adjacent federations differing by the addition or removal of one client (Poisson subsampling model):
   $$\Delta_2\left(\sum_{i \in S_t} \Delta \tilde{w}_i^t\right) = C$$

3. **Central Noise Addition on the Averaged Update**:
   Calibrated Gaussian noise with variance $\sigma^2 C^2 I$ is added to the unweighted sum:
   $$\Delta_{\text{sum}}^{\text{priv}} = \sum_{i \in S_t} \Delta \tilde{w}_i^t + \mathcal{N}(0, \sigma^2 C^2 I)$$
   When dividing by cohort size $m$ to compute the parameter step $\bar{\Delta}^t$:
   $$\bar{\Delta}^t = \frac{1}{m} \sum_{i \in S_t} \Delta \tilde{w}_i^t + \mathcal{N}\left(0, \frac{\sigma^2 C^2}{m^2} I\right)$$
   where $\sigma$ is the noise multiplier passed to the accountant.

### 3.3 Composition & Epsilon Accounting
Privacy guarantees across $T$ communication rounds are tracked using Rényi Differential Privacy (RDP) (Mironov, CSF 2017) with parameters:
- Steps: $T$ communication rounds
- Subsampling ratio: $q = m / K$
- Noise multiplier: $\sigma$
- Target failure probability: $\delta = 10^{-5}$

### 3.4 Independent Cross-Verification against Google `dp_accounting`
To avoid circular validation (e.g. verifying an Opacus wrapper using Opacus's own output), FLaaS cross-checks every $(\varepsilon, \delta)$ guarantee against Google's independent `dp_accounting` library (v0.6.0, `dp_accounting.rdp.RdpAccountant`). Both libraries evaluate numerical integrations over the mixture of two Gaussians across discrete Rényi order grids, agreeing to within $< 0.05\%$ relative difference.

