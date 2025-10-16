import random
import numpy as np

class TrafficEnv:
    '''
    Simplified intersection:
    - Two directions: NS and EW
    - Phase 0: NS green (cars from NS pass), Phase 1: EW green
    - Discrete-time steps; each step new cars arrive with probability p_arrival
    - Passing rate: up to `pass_per_step` cars leave from the green direction
    '''
    def __init__(self, max_queue=10, p_arrival=0.3, pass_per_step=1, seed=None):
        self.max_queue = max_queue
        self.p_arrival = p_arrival
        self.pass_per_step = pass_per_step
        self.rng = random.Random(seed)
        self.reset()
    def reset(self):
        self.ns_queue = 0
        self.ew_queue = 0
        self.phase = 0  # 0 = NS green, 1 = EW green
        self.t = 0
        return self._get_state()
    def step(self, action):
        # action: 0 -> keep phase, 1 -> switch phase
        if action == 1:
            self.phase = 1 - self.phase
        # arrivals
        if self.rng.random() < self.p_arrival:
            self.ns_queue = min(self.max_queue, self.ns_queue + 1)
        if self.rng.random() < self.p_arrival:
            self.ew_queue = min(self.max_queue, self.ew_queue + 1)
        # departures
        if self.phase == 0:
            departed = min(self.pass_per_step, self.ns_queue)
            self.ns_queue -= departed
        else:
            departed = min(self.pass_per_step, self.ew_queue)
            self.ew_queue -= departed
        # reward: negative of total queue lengths (penalize long queues)
        reward = - (self.ns_queue + self.ew_queue)
        self.t += 1
        done = False
        return self._get_state(), reward, done, {}
    def _get_state(self):
        # discretized state tuple
        return (self.ns_queue, self.ew_queue, self.phase)
