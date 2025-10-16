import numpy as np
import os

class QAgent:
    def __init__(self, max_q=10, n_actions=2, alpha=0.1, gamma=0.99, epsilon=1.0, min_epsilon=0.05, decay=0.9995):
        self.max_q = max_q
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        # Q-table shape: (ns_q+1, ew_q+1, phases, actions)
        self.Q = np.zeros((max_q+1, max_q+1, 2, n_actions))
    def state_to_idx(self, state):
        ns, ew, phase = state
        ns = min(self.max_q, max(0, ns))
        ew = min(self.max_q, max(0, ew))
        phase = 0 if phase==0 else 1
        return (ns, ew, phase)
    def choose_action(self, state):
        ns, ew, phase = self.state_to_idx(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        qvals = self.Q[ns, ew, phase]
        return int(np.argmax(qvals))
    def learn(self, state, action, reward, next_state):
        s_idx = self.state_to_idx(state)
        ns_idx = self.state_to_idx(next_state)
        q = self.Q[s_idx + (action,)]
        q_next = np.max(self.Q[ns_idx])
        self.Q[s_idx + (action,)] = q + self.alpha * (reward + self.gamma * q_next - q)
        # decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
    def save(self, path):
        np.save(path, self.Q)
    def load(self, path):
        self.Q = np.load(path)
