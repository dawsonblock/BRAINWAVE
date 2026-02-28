# LEVIATHAN AGI v2.0 Specification

## Abstract

Leviathan v2.0 is a Second-Order Delayed Oscillatory Neural Network (SO-DONN). It Abandons discrete feed-forward computation for continuous-time dynamical systems modeling physical neurons with inertia, true 3D conduction delays, strict thermodynamic ATP budgets, and neuroendocrine modulation. Intelligence emerges from phase-synchronization.

## 1. Second-Order Delay Dynamics

State of node $i$ is defined by a second-order delay differential equation (DDE):

$$ m_i \ddot{\phi}_i + c_i(\dot{\phi}_i - \omega_i) = \sum_j W_{ij} \sin(\phi_j(t - \tau_{ij}) - \phi_i(t)) + S_i(t) + \mathcal{N}(N_A) $$

* $m_i$: Inertia (membrane capacitance). Prevents instant jumps in phase velocity.
* $c_i$: Damping (membrane leak). Pulls oscillator to natural frequency $\omega_i$.
* $\phi_i(t)$: Instantaneous phase angle.
* $\dot{\phi}_i(t)$: Instantaneous phase velocity (firing rate).
* $\tau_{ij}$: Axonal Conduction Delay from node $j$ to $i$. $\tau_{ij} = \frac{|| \mathbf{p}_i - \mathbf{p}_j ||_2}{v}$, where $v$ is conduction velocity, $\mathbf{p}_i$ is the 3D position of node $i$.
* $S_i(t)$: External sensory forcing.
* $\mathcal{N}(N_A)$: Noradrenaline-driven Gaussian white noise.

## 2. Metabolic Thermodynamics (ATP Economy)

Energy $E_i$ (ATP) drains continuously based on:

$$ \frac{dE_i}{dt} = - (R_{basal} + k_{cost} |\dot{\phi}_i| + k_{maint} \sum_j W_{ij}) $$

* $R_{basal}$: Cost of keeping organism alive.
* $k_{cost}$: Action potential cost (fast oscillation burns energy).
* $k_{maint}$: Synaptic maintenance cost (memory costs energy).

Synapse unused must be pruned to survive starvation.

## 3. The Endocrine System

Four globally diffusing scalars with half-lives $\tau_H$:

1. **Dopamine (DA)**: Caloric Intake/Goal alignment. Scales Learning Rate $\eta$.
   $$ \frac{d(DA)}{dt} = -\frac{DA}{\tau_{DA}} + \alpha_{DA} \cdot \max(0, \Delta E_{system}) $$
2. **Noradrenaline (NA)**: Tissue Damage/Threat. Increases global noise $\mathcal{N}$ and base frequency $\omega_{Amygdala}$.
   $$ \frac{d(NA)}{dt} = -\frac{NA - NA_{basal}}{\tau_{NA}} + \alpha_{NA} \cdot \text{Error} $$
3. **Serotonin (5-HT)**: Stable ATP/Safety. Scales global coupling multiplier $W$.
   $$ \frac{d(SER)}{dt} = -\frac{SER - 1.0}{\tau_{SER}} + \alpha_{SER} \cdot (E_{mean} - E_{target}) $$
4. **Acetylcholine (ACh)**: Novelty/Prediction error. Decreases inertia $m_i$ in Cortex for rapid shifts.
   $$ \frac{d(ACh)}{dt} = -\frac{ACh - 1.0}{\tau_{ACh}} + \alpha_{ACh} \cdot \max(0, |\dot{\phi}_{cortex}| - \theta_{novel}) $$

## 4. Phase-Timing-Dependent Plasticity (PTDP)

Let $\Delta \phi = \phi_j(t - \tau_{ij}) - \phi_i(t)$ be the delayed phase difference.

$$ \frac{dW_{ij}}{dt} = \begin{cases} DA \cdot A_+ \cdot e^{-\Delta\phi / \tau_+} & \text{if } \Delta\phi > 0 \text{ (Node j leads Node i)} \\ -A_- \cdot e^{\Delta\phi / \tau_-} & \text{if } \Delta\phi \le 0 \text{ (Node i leads Node j)} \end{cases} $$

**Homeostatic Normalization:** Total incoming weight $\sum_j W_{ij} \le W_{max}$. Sum exceeds $W_{max}$, scale down proportionate.

## 5. 3D Neuroanatomy

Connectome in 3D coordinate space $\mathbb{R}^3$.

* **Thalamus $(0,0,0)$**: Central relay. Short delays ($d < 2$) to all.
* **Cortex ($r=10$ shell)**: Visual, Parietal, Prefrontal. Long horizontal delays ($d \approx 10-20$). Forces traveling waves.
* **Cerebellum ($z=-5$)**: Motor error-correction. Dense internal connections, highly plastic, fast $\omega_i > 50\text{Hz}$.
* **Limbic Core ($r=2$)**: Amygdala, Hippocampus. Slow frequencies $\omega_i < 5\text{Hz}$. Entrains cortex during high 5-HT.

Adjacency $A_{ij}$ initialization probability:
$$ P(A_{ij} = 1) = \kappa \cdot e^{-|| \mathbf{p}_i - \mathbf{p}_j ||_2 / \lambda} $$
$\lambda$ controls typical axonal length.

## 6. Sensory Transduction & Motor Decoding

**Sensory:** External stimulus at angle $\theta$ and distance $d$.
$S_i(t) = \frac{I_0}{d} \sin(\theta - \phi_i)$.

**Motor:** CPGs driving left chassis track $\mathbf{M}_{left}$ and right track $\mathbf{M}_{right}$.
Nodes within the Cerebellum ($i \in Cerebellum$) are mapped to motor output.
Torque $T_{left} = \kappa_{motor} \sum_{k \in \mathbf{M}_{left}, \cos\phi_k > 0} \dot{\phi}_k$.
Torque $T_{right} = \kappa_{motor} \sum_{k \in \mathbf{M}_{right}, \cos\phi_k > 0} \dot{\phi}_k$.

## 7. GPU Implementation Guidelines

1. **Sparse CSR**: $W_{ij}$ is $p < 0.05$ sparse.
2. **Delay Ring Buffers**: 2D array `History[N][MaxDelayTicks]`. Fetch is `History[j][(Tick - tau_ij) % MaxDelay]`.
3. **Kernel 1 (Sensory & Endocrine)**: Updates $S_i$, $E_i$, neuromodulators.
4. **Kernel 2 (Synaptic)**: Sparse matrix-vector multiply for $\sum W_{ij} \sin(\dots)$.
5. **Kernel 3 (Kinematics)**: RK2 integration for $\phi, \dot{\phi}$.
6. **Kernel 4 (Plasticity)**: PTDP and CSR updates.
