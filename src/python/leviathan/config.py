# --- 1. Physics & Delays ---
CONDUCTION_VELOCITY = 10.0  # arbitrary distance units per second
DT = 0.01  # Integration time step (seconds)
MAX_DELAY_SEC = 2.0  # Maximum time to store in ring buffers
MAX_DELAY_TICKS = int(MAX_DELAY_SEC / DT)

# --- 2. Neural Properties ---
DEFAULT_INERTIA = 1.0  # m_i
DEFAULT_DAMPING = 0.5  # c_i
SYNAPTIC_W_MAX = 5.0  # Homeostatic max sum of incoming weights

# --- 3. Endocrine & Plasticity (PTDP) ---
DA_LEARN_MOD = 1.0  # Dopamine learning rate scale
A_PLUS = 0.1  # PTDP LTP max amplitude
A_MINUS = 0.12  # PTDP LTD max amplitude (LTD slightly stronger)
TAU_PLUS = 0.05  # LTP time constant
TAU_MINUS = 0.05  # LTD time constant

# Dynamic Endocrine Constants
TAU_DA = 5.0  # Dopamine (DA) - Exploration & Reward
ALPHA_DA = 0.5
DA_BASAL = 0.1

# Noradrenaline (NA) - Arousal & Novelty
TAU_NA = 3.0
ALPHA_NA = 0.8
NA_BASAL = 0.1
NOVELTY_THRESHOLD = 2.0

# Serotonin (SER) - Stability & Coupling Strength
TAU_SER = 10.0
ALPHA_SER = 0.3
SER_BASAL = 1.0

# Acetylcholine (ACH) - Attention & Signal Relay
TAU_ACH = 2.0
ALPHA_ACH = 0.6
ACH_BASAL = 0.5

# --- EMBODIED ENVIRONMENT (PHASE 7) ---
ATP_FOOD_REWARD = 500.0
DA_REWARD_SPIKE = 2.0

SENSORY_GAIN = 80.0
MOTOR_GAIN = 10.0
WALL_GAIN = 48.0

ATP_TARGET = 800.0
THETA_NOVEL = 5.0  # rad/s threshold for novelty detection

# --- 4. Thermodynamics (ATP Economy) ---
INITIAL_ATP = 1000.0
R_BASAL = 0.1  # Base ATP drain per second
K_COST = 0.05  # ATP cost per rad/s phase velocity
K_MAINT = 0.02  # ATP cost per unit of synaptic weight
ATP_DEATH_THRESHOLD = 0.0  # If ATP hits 0, neuron "dies" (or prunes synapses)

# --- 5. Brain Regions & Topology ---
# Defines [Radius, z_range, base_freq_mean, base_freq_std]
REGIONS = {
    "thalamus": {"r_min": 0.0, "r_max": 1.0, "z": (0, 0), "omega": (20.0, 2.0)},
    "limbic": {"r_min": 1.0, "r_max": 3.0, "z": (-2, 2), "omega": (4.0, 1.0)},
    "cortex_v": {"r_min": 8.0, "r_max": 10.0, "z": (0, 5), "omega": (10.0, 2.0)},
    "cortex_p": {"r_min": 8.0, "r_max": 10.0, "z": (-5, 0), "omega": (10.0, 2.0)},
    "cerebellum": {"r_min": 2.0, "r_max": 5.0, "z": (-8, -5), "omega": (50.0, 5.0)},
}

TOPOLOGY_LAMBDA = 5.0  # decay constant for connection probability
