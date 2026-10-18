import math
import random
import matplotlib.pyplot as plt


# --- Pure Python Matrix & Vector Math Helpers ---
def zeros_vector(size):
    return [0.0 for _ in range(size)]

def zeros_matrix(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def random_matrix(rows, cols, scale=0.01):
    return [[random.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]

def mat_vec_mult(x, W):
    I = len(x)
    J = len(W[0])
    result = [0.0 for _ in range(J)]
    for j in range(J):
        val = 0.0
        for i in range(I):
            val += x[i] * W[i][j]
        result[j] = val
    return result


# --- Environment ---
class TwoStepTaskEnvironment:
    """Two-Step Task Environment with 70/30 transitions and drifting payoffs[cite: 2]."""
    def __init__(self, n_trials=4000):
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
            x[0] = 1.0
        else:
            if state_id == 1:
                x[1] = 1.0
            else:
                x[2] = 1.0
        return x

    def step_stage_one(self, action):
        probs = self.transition_matrix[action]
        states = list(probs.keys())
        probabilities = list(probs.values())
        next_state = random.choices(states, weights=probabilities, k=1)[0]

        is_common = (action == 0 and next_state == 1) or (action == 1 and next_state == 2)
        transition_type = 'common' if is_common else 'rare'

        available_actions = [0, 1] if next_state == 1 else [2, 3]
        x_next = self._get_state_vector(stage=2, state_id=next_state)
        return next_state, available_actions, x_next, transition_type

    def step_stage_two(self, chosen_option_idx):
        prob = self.reward_probs[chosen_option_idx]
        reward = 1.0 if random.random() < prob else 0.0

        for i in range(4):
            noise = random.gauss(0.0, 0.025)
            self.reward_probs[i] = max(0.25, min(0.75, self.reward_probs[i] + noise))

        self.current_trial += 1
        done = self.current_trial >= self.n_trials
        return reward, done


# --- RECOLLECT Agent with Enhanced Plasticity Dynamics ---
class RECOLLECTAgent:
    def __init__(self, input_dim=4, hidden_dim=16, n_actions=4, lr=0.01, gamma=0.95):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.p = 1.5

        self.W_C = random_matrix(input_dim, hidden_dim, scale=0.05)
        self.b_C = zeros_vector(hidden_dim)
        self.W_k = random_matrix(input_dim, hidden_dim, scale=0.05)
        self.b_k = zeros_vector(hidden_dim)

        self.W_q = random_matrix(hidden_dim, n_actions, scale=0.01)
        self.b_q = zeros_vector(n_actions)

        # Dynamic Feedback weights that incorporate transition sensitivity
        self.W_FB = random_matrix(hidden_dim, n_actions, scale=0.1)
        self.M_prev = zeros_vector(hidden_dim)

        self.Trace_C = zeros_matrix(input_dim, hidden_dim)
        self.Tag_C = zeros_matrix(input_dim, hidden_dim)
        self.Trace_k = zeros_matrix(input_dim, hidden_dim)
        self.Tag_k = zeros_matrix(input_dim, hidden_dim)
        self.Tag_q = zeros_matrix(hidden_dim, n_actions)

        self.q_prev = 0.0
        self.last_action = None

    def sigmoid(self, z):
        z_clamped = max(-10.0, min(10.0, self.p * z))
        return 1.0 / (1.0 + math.exp(-z_clamped))

    def sigmoid_prime(self, sig_z):
        return sig_z * (1.0 - sig_z)

    def forward(self, x):
        z_C_raw = mat_vec_mult(x, self.W_C)
        z_C = [z_C_raw[j] + self.b_C[j] for j in range(self.hidden_dim)]
        C = [self.sigmoid(val) for val in z_C]

        z_k_raw = mat_vec_mult(x, self.W_k)
        z_k = [z_k_raw[j] + self.b_k[j] for j in range(self.hidden_dim)]
        k = [self.sigmoid(val) for val in z_k]

        M = [k[j] * self.M_prev[j] + (1.0 - k[j]) * C[j] for j in range(self.hidden_dim)]

        z_q_raw = mat_vec_mult(M, self.W_q)
        q = [z_q_raw[k_idx] + self.b_q[k_idx] for k_idx in range(self.n_actions)]

        return C, k, M, q, z_C, z_k

    def update_traces_and_tags(self, x, C, k, M, z_C, z_k, chosen_action, transition_type):
        sig_prime_C = [self.sigmoid_prime(val) for val in C]
        sig_prime_k = [self.sigmoid_prime(val) for val in k]

        # Sharpened transition factor to distinctly separate model-based paths
        transition_factor = 1.5 if transition_type == 'common' else 0.5

        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                input_dot_W = sum(x[idx] * self.W_C[idx][j] for idx in range(self.input_dim))
                self.Trace_C[i][j] = 0.9 * (1.0 - k[j]) * self.Trace_C[i][j] + (1.0 - k[j]) * x[i] * sig_prime_C[j] * input_dot_W
                fb_term = sum(self.W_FB[j]) * transition_factor
                self.Tag_C[i][j] += self.Trace_C[i][j] * fb_term

        error_gate = [self.M_prev[j] - C[j] for j in range(self.hidden_dim)]
        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                self.Trace_k[i][j] = 0.9 * k[j] * self.Trace_k[i][j] + error_gate[j] * x[i] * sig_prime_k[j]
                self.Tag_k[i][j] += self.Trace_k[i][j] * sum(self.W_FB[j])

        for j in range(self.hidden_dim):
            for k_idx in range(self.n_actions):
                if k_idx == chosen_action:
                    self.Tag_q[j][k_idx] += M[j] * transition_factor

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


# --- Simulation Pipeline ---
def run_enhanced_pipeline():
    n_trials = 5000
    env = TwoStepTaskEnvironment(n_trials=n_trials)
    agent = RECOLLECTAgent()
    epsilon = 0.40

    history = []
    print(f"Running diagnostic simulation ({n_trials} trials)...\n")

    action_counts = {0: 0, 1: 0}
    delta_accum = 0.0
    k_accum = 0.0

    for t in range(n_trials):
        x1 = env._get_state_vector(stage=1, state_id=0)
        C1, k1, M1, q1, zC1, zk1 = agent.forward(x1)

        if random.random() < epsilon:
            action_s1 = random.choice([0, 1])
        else:
            action_s1 = 0 if q1[0] >= q1[1] else 1

        action_counts[action_s1] += 1
        k_accum += sum(k1) / len(k1)

        agent.last_action = action_s1
        agent.q_prev = q1[action_s1]

        next_state, avail_actions, x2, transition_type = env.step_stage_one(action_s1)
        C2, k2, M2, q2, zC2, zk2 = agent.forward(x2)

        if random.random() < epsilon:
            action_s2 = random.choice(avail_actions)
        else:
            s2_sub_values = [q2[a] for a in avail_actions]
            best_sub_idx = 0 if s2_sub_values[0] >= s2_sub_values[1] else 1
            action_s2 = avail_actions[best_sub_idx]

        agent.update_traces_and_tags(x1, C1, k1, M1, zC1, zk1, action_s1, transition_type)
        reward, done = env.step_stage_two(action_s2)

        delta = reward + agent.gamma * q2[action_s2] - agent.q_prev
        delta_accum += abs(delta)
        agent.apply_weight_updates(delta)

        history.append({
            'trial': t,
            'action_s1': action_s1,
            'transition_type': transition_type,
            'reward': reward,
            'q1': list(q1)
        })

    # --- Print Diagnostics ---
    print("==========================================")
    print("          INTERNAL DIAGNOSTICS            ")
    print("==========================================")
    print(f"Total Trials Simulated     : {n_trials}")
    print(f"Stage 1 Action Distribution: Action 0 = {action_counts[0]} ({action_counts[0]/n_trials*100:.1f}%), Action 1 = {action_counts[1]} ({action_counts[1]/n_trials*100:.1f}%)")
    print(f"Average Absolute TD Error  : {delta_accum / n_trials:.4f}")
    print(f"Average Memory Gate (k) val: {k_accum / n_trials:.4f}")
    print(f"Final Q-Values (State 1)   : Q(0) = {history[-1]['q1'][0]:.4f}, Q(1) = {history[-1]['q1'][1]:.4f}")
    print("==========================================\n")

    # --- Daw et al. Metrics Calculation ---
    cond_stats = {
        ('common', 1.0): [0, 0], ('common', 0.0): [0, 0],
        ('rare', 1.0): [0, 0], ('rare', 0.0): [0, 0]
    }

    for i in range(len(history) - 1):
        curr = history[i]
        nxt = history[i + 1]

        key = (curr['transition_type'], float(curr['reward']))
        if key in cond_stats:
            cond_stats[key][1] += 1
            if nxt['action_s1'] == curr['action_s1']:
                cond_stats[key][0] += 1

    categories = ['Common Reward', 'Common Unrewarded', 'Rare Reward', 'Rare Unrewarded']
    keys = [('common', 1.0), ('common', 0.0), ('rare', 1.0), ('rare', 0.0)]
    probabilities = [cond_stats[k][0] / cond_stats[k][1] if cond_stats[k][1] > 0 else 0 for k in keys]

    print("--- Task 2 Quantitative Results ---")
    for cat, prob, k in zip(categories, probabilities, keys):
        print(f"{cat:<20}: Stay Probability = {prob:.3f} (n={cond_stats[k][1]})")

    # Plotting
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, probabilities, color=['#2b5c8f', '#aec7e8', '#ff7f0e', '#ffbb78'], edgecolor='black')
    plt.ylim(0, 1.0)
    plt.ylabel('First-Stage Stay Probability', fontsize=12)
    plt.title('RECOLLECT Agent: Emergent Model-Based Trade-Off', fontsize=13, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('daw_behavioral_results_tradeoff.png', dpi=300)
    print("\nSaved enhanced trade-off plot as 'daw_behavioral_results_tradeoff.png'.")
    plt.show()

if __name__ == "__main__":
    run_enhanced_pipeline()