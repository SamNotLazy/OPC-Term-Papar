import math
import random
import unittest


# --- Pure Python Matrix & Vector Math Helpers ---
def zeros_vector(size):
    return [0.0 for _ in range(size)]

def zeros_matrix(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def random_matrix(rows, cols, scale=0.1):
    return [[random.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]

def mat_vec_mult(x, W):
    """
    Multiplies vector x (length I) by matrix W (I x J).
    Returns vector of length J.
    """
    I = len(x)
    J = len(W[0])
    result = [0.0 for _ in range(J)]
    for j in range(J):
        val = 0.0
        for i in range(I):
            val += x[i] * W[i][j]
        result[j] = val
    return result

def vec_add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

def vec_sub(v1, v2):
    return [a - b for a, b in zip(v1, v2)]

def vec_mul(v1, v2):
    return [a * b for a, b in zip(v1, v2)]

def scalar_mult(scalar, matrix_or_vec):
    if isinstance(matrix_or_vec[0], list):
        return [[scalar * val for val in row] for row in matrix_or_vec]
    else:
        return [scalar * val for val in matrix_or_vec]


# --- Environment ---
class TwoStepTaskEnvironment:
    """Two-Step Task Environment with 70/30 transitions and drifting payoffs."""
    def __init__(self, n_trials=201):
        self.n_trials = n_trials
        self.current_trial = 0
        self.reward_probs = [0.5, 0.5, 0.5, 0.5]
        self.transition_matrix = {
            0: {1: 0.7, 2: 0.3},
            1: {2: 0.7, 1: 0.3}
        }

    def reset(self):
        self.current_trial = 0
        self.reward_probs = [0.5, 0.5, 0.5, 0.5]
        return self._get_state_vector(stage=1, state_id=0)

    def _get_state_vector(self, stage, state_id):
        x = [0.0, 0.0, 0.0, 0.0]
        if stage == 1:
            x[0] = 1.0  # Stage 1 indicator
        else:
            if state_id == 1:
                x[1] = 1.0  # Second-stage State 1 indicator
            else:
                x[2] = 1.0  # Second-stage State 2 indicator
        return x

    def step_stage_one(self, action):
        probs = self.transition_matrix[action]
        states = list(probs.keys())
        probabilities = list(probs.values())
        next_state = random.choices(states, weights=probabilities, k=1)[0]

        available_actions = [0, 1] if next_state == 1 else [2, 3]
        x_next = self._get_state_vector(stage=2, state_id=next_state)
        return next_state, available_actions, x_next

    def step_stage_two(self, chosen_option_idx):
        prob = self.reward_probs[chosen_option_idx]
        reward = 1 if random.random() < prob else 0

        # Gaussian random walk drift with reflecting boundaries at 0.25 and 0.75
        for i in range(4):
            noise = random.gauss(0.0, 0.025)
            self.reward_probs[i] = max(0.25, min(0.75, self.reward_probs[i] + noise))

        self.current_trial += 1
        done = self.current_trial >= self.n_trials
        return reward, done


# --- RECOLLECT Agent (Pure Python Lists) ---
class RECOLLECTAgent:
    """
    RECOLLECT architecture using Light-GRUs, synaptic tags/traces,
    and three-factor local learning rules implemented entirely with Python lists.
    """
    def __init__(self, input_dim=4, hidden_dim=8, n_actions=4, lr=0.01, gamma=0.95):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.p = 2.0  # Sigmoid slope parameter

        self.W_C = random_matrix(input_dim, hidden_dim, scale=0.1)
        self.b_C = zeros_vector(hidden_dim)

        self.W_k = random_matrix(input_dim, hidden_dim, scale=0.1)
        self.b_k = zeros_vector(hidden_dim)

        self.W_q = random_matrix(hidden_dim, n_actions, scale=0.1)
        self.b_q = zeros_vector(n_actions)

        self.W_FB = [[0.1 for _ in range(n_actions)] for _ in range(hidden_dim)]
        self.M_prev = zeros_vector(hidden_dim)

        self.Trace_C = zeros_matrix(input_dim, hidden_dim)
        self.Tag_C = zeros_matrix(input_dim, hidden_dim)

        self.Trace_k = zeros_matrix(input_dim, hidden_dim)
        self.Tag_k = zeros_matrix(input_dim, hidden_dim)

        self.Tag_q = zeros_matrix(hidden_dim, n_actions)

        self.q_prev = 0.0
        self.last_action = None

    def sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-self.p * z))

    def sigmoid_prime(self, sig_z):
        return self.p * sig_z * (1.0 - sig_z)

    def forward(self, x):
        # Candidate memory C_j
        z_C_raw = mat_vec_mult(x, self.W_C)
        z_C = [z_C_raw[j] + self.b_C[j] for j in range(self.hidden_dim)]
        C = [self.sigmoid(val) for val in z_C]

        # Gating units k_j
        z_k_raw = mat_vec_mult(x, self.W_k)
        z_k = [z_k_raw[j] + self.b_k[j] for j in range(self.hidden_dim)]
        k = [self.sigmoid(val) for val in z_k]

        # Memory units M_j
        M = [k[j] * self.M_prev[j] + (1.0 - k[j]) * C[j] for j in range(self.hidden_dim)]

        # Q-values q_k
        z_q_raw = mat_vec_mult(M, self.W_q)
        z_q = [z_q_raw[k_idx] + self.b_q[k_idx] for k_idx in range(self.n_actions)]
        q = [self.sigmoid(val) for val in z_q]

        return C, k, M, q, z_C, z_k

    def update_traces_and_tags(self, x, C, k, M, z_C, z_k, chosen_action):
        sig_prime_C = [self.sigmoid_prime(val) for val in C]
        sig_prime_k = [self.sigmoid_prime(val) for val in k]

        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                input_dot_W = sum(x[idx] * self.W_C[idx][j] for idx in range(self.input_dim))
                self.Trace_C[i][j] = k[j] * self.Trace_C[i][j] + (1.0 - k[j]) * x[i] * sig_prime_C[j] * input_dot_W
                fb_term = sum(self.W_FB[j])
                self.Tag_C[i][j] += self.Trace_C[i][j] * fb_term

        error_gate = [self.M_prev[j] - C[j] for j in range(self.hidden_dim)]
        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                self.Trace_k[i][j] = k[j] * self.Trace_k[i][j] + error_gate[j] * x[i] * sig_prime_k[j]
                self.Tag_k[i][j] += self.Trace_k[i][j] * sum(self.W_FB[j])

        for j in range(self.hidden_dim):
            for k_idx in range(self.n_actions):
                if k_idx == chosen_action:
                    self.Tag_q[j][k_idx] += M[j]

        self.M_prev = list(M)

    def apply_weight_updates(self, delta):
        for j in range(self.hidden_dim):
            for k_idx in range(self.n_actions):
                self.W_q[j][k_idx] += self.lr * delta * self.Tag_q[j][k_idx]

        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                self.W_C[i][j] += self.lr * delta * self.Tag_C[i][j]
                self.W_k[i][j] += self.lr * delta * self.Tag_k[i][j]

        # Reset tags
        self.Tag_q = zeros_matrix(self.hidden_dim, self.n_actions)
        self.Tag_C = zeros_matrix(self.input_dim, self.hidden_dim)
        self.Tag_k = zeros_matrix(self.input_dim, self.hidden_dim)
# --- Example Integration Simulation Loop ---
if __name__ == "__main__":
    env = TwoStepTaskEnvironment(n_trials=5)
    agent = RECOLLECTAgent()

    print("Starting RECOLLECT Two-Step Task Simulation (Pure Python Architecture)...\n")

    for t in range(3):
        print(f"=== Trial {t + 1} ===")
        x1 = env._get_state_vector(stage=1, state_id=0)

        # Forward pass for Stage 1
        C1, k1, M1, q1, zC1, zk1 = agent.forward(x1)

        # Choose action via greedy selection (argmax) on q1 values (indices 0 and 1)
        action_s1 = 0 if q1[0] > q1[1] else 1
        agent.last_action = action_s1
        agent.q_prev = q1[action_s1]

        print(f"  [Stage 1] Input Vector x1: {x1}")
        print(f"  [Stage 1] Gating Units k1 (sample): {[round(val, 3) for val in k1[:4]]}")
        print(f"  [Stage 1] Memory Units M1 (sample): {[round(val, 3) for val in M1[:4]]}")
        print(f"  [Stage 1] Action Q-values q1: {[round(val, 4) for val in q1]}")
        print(f"  [Stage 1] Chosen Action: Option {action_s1}")

        # Step environment stage 1
        next_state, avail_actions, x2 = env.step_stage_one(action_s1)
        print(f"  -> Transitioned to Second-Stage State {next_state} (Available actions: {avail_actions})")

        # Forward pass for Stage 2
        C2, k2, M2, q2, zC2, zk2 = agent.forward(x2)

        # Choose action at Stage 2 from available actions
        s2_sub_values = [q2[a] for a in avail_actions]
        best_sub_idx = 0 if s2_sub_values[0] > s2_sub_values[1] else 1
        action_s2 = avail_actions[best_sub_idx]

        print(f"  [Stage 2] Action Q-values q2: {[round(val, 4) for val in q2]}")
        print(f"  [Stage 2] Chosen Action: Option {action_s2}")

        # Update local synaptic traces and tags based on trial history
        agent.update_traces_and_tags(x1, C1, k1, M1, zC1, zk1, action_s1)

        # Receive reward and run environment drift (reflecting bounds [0.25, 0.75])
        reward, done = env.step_stage_two(action_s2)
        print(f"  [Outcome] Reward Received: £{reward} | True Reward Probs: {[round(p, 3) for p in env.reward_probs]}")

        # Compute Global RPE (delta) using SARSA TD rule
        current_q = q2[action_s2]
        delta = reward + agent.gamma * current_q - agent.q_prev

        # Track synaptic weight sample before update to verify local plasticity
        w_sample_before = agent.W_q[0][action_s1]

        # Apply local three-factor weight updates using global RPE and tags
        agent.apply_weight_updates(delta)

        w_sample_after = agent.W_q[0][action_s1]
        print(f"  [Plasticity] Computed Global RPE (delta): {delta:.4f}")
        print(f"  [Plasticity] Synaptic Weight W_q[0][{action_s1}] updated locally: {w_sample_before:.4f} -> {w_sample_after:.4f}\n")