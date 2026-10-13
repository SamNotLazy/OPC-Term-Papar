# --- Unit Test Suite ---
class TestTwoStepRECOLLECT(unittest.TestCase):

    def setUp(self):
        self.env = TwoStepTaskEnvironment(n_trials=10)
        self.agent = RECOLLECTAgent(input_dim=4, hidden_dim=8, n_actions=4)

    def test_environment_reset(self):
        state_x = self.env.reset()
        self.assertEqual(self.env.current_trial, 0)
        self.assertTrue(all(p == 0.5 for p in self.env.reward_probs))
        self.assertEqual(len(state_x), 4)
        self.assertEqual(state_x[0], 1.0)

    def test_environment_transitions(self):
        self.env.reset()
        next_state, avail_actions, state_x2 = self.env.step_stage_one(action=0)
        self.assertIn(next_state, [1, 2])
        self.assertEqual(len(state_x2), 4)
        if next_state == 1:
            self.assertEqual(avail_actions, [0, 1])
        else:
            self.assertEqual(avail_actions, [2, 3])

    def test_reward_drifting_boundaries(self):
        self.env.reset()
        for _ in range(50):
            self.env.step_stage_two(chosen_option_idx=0)
        self.assertTrue(all(p >= 0.25 and p <= 0.75 for p in self.env.reward_probs))

    def test_recollect_forward_pass(self):
        state_x = self.env.reset()
        C, k, M, q, z_C, z_k = self.agent.forward(state_x)

        self.assertEqual(len(C), 8)
        self.assertEqual(len(k), 8)
        self.assertEqual(len(M), 8)
        self.assertEqual(len(q), 4)

        self.assertTrue(all(0.0 <= val <= 1.0 for val in C))
        self.assertTrue(all(0.0 <= val <= 1.0 for val in k))
        self.assertTrue(all(0.0 <= val <= 1.0 for val in q))

    def test_local_plasticity_updates(self):
        initial_w_q_val = self.agent.W_q[0][0]

        state_x1 = self.env.reset()
        C1, k1, M1, q1, zC1, zk1 = self.agent.forward(state_x1)
        action_s1 = 0

        next_state, avail_actions, state_x2 = self.env.step_stage_one(action_s1)
        C2, k2, M2, q2, zC2, zk2 = self.agent.forward(state_x2)
        action_s2 = avail_actions[0]

        self.agent.update_traces_and_tags(state_x1, C1, k1, M1, zC1, zk1, action_s1)

        delta = 1.0
        self.agent.apply_weight_updates(delta)

        # Verify weights updated and tags reset
        self.assertNotEqual(self.agent.W_q[0][0], initial_w_q_val)
        self.assertTrue(all(all(val == 0.0 for val in row) for row in self.agent.Tag_q))


if __name__ == '__main__':
    unittest.main()