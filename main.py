from train import train, evaluate
from environment import TrafficEnv
from agent import QAgent
import matplotlib.pyplot as plt
import numpy as np
import os

def run_pipeline():
    # 1) Evaluate random (untrained) agent (epsilon=1 -> random)
    env = TrafficEnv(seed=123)
    rand_agent = QAgent()
    rand_agent.epsilon = 1.0
    before = []
    for _ in range(50):
        s = env.reset()
        rsum = 0
        for _ in range(100):
            a = rand_agent.choose_action(s)
            s, r, _, _ = env.step(a)
            rsum += r
        before.append(rsum)
    avg_before = np.mean(before)
    print(f"Average episode reward (random agent): {avg_before:.2f}")
    # 2) Train agent
    agent, env, rewards = train(episodes=2000, max_steps=100)
    # 3) Evaluate trained agent (epsilon has decayed)
    env = TrafficEnv(seed=999)
    after = []
    for _ in range(50):
        s = env.reset()
        rsum = 0
        for _ in range(100):
            a = agent.choose_action(s)
            s, r, _, _ = env.step(a)
            rsum += r
        after.append(rsum)
    avg_after = np.mean(after)
    print(f"Average episode reward (trained agent): {avg_after:.2f}")
    # 4) Save comparison plot
    os.makedirs('results', exist_ok=True)
    plt.figure()
    plt.bar(['Before (random)', 'After (trained)'], [avg_before, avg_after])
    plt.ylabel('Average Episode Reward')
    plt.title('Before vs After Training')
    plt.savefig('results/avg_wait_before_after.png')
    plt.close()
    print('Results saved to results/ directory.')

if __name__ == '__main__':
    run_pipeline()
