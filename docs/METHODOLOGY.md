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
