import numpy as np
import matplotlib.pyplot as plt
from environment import TrafficEnv
from agent import QAgent
import os

def evaluate(agent, env, episodes=50, max_steps=200):
    total_rewards = []
    for _ in range(episodes):
        s = env.reset()
        rsum = 0
        for _ in range(max_steps):
            a = agent.choose_action(s)
            s, r, _, _ = env.step(a)
            rsum += r
        total_rewards.append(rsum)
    return np.mean(total_rewards)

def train(episodes=2000, max_steps=100):
    env = TrafficEnv(seed=42)
    agent = QAgent()
    rewards = []
    for ep in range(episodes):
        s = env.reset()
        r_ep = 0
        for step in range(max_steps):
            a = agent.choose_action(s)
            s2, r, done, _ = env.step(a)
            agent.learn(s, a, r, s2)
            s = s2
            r_ep += r
        rewards.append(r_ep)
        if (ep+1) % 200 == 0:
            print(f"Episode {ep+1}/{episodes}, reward (last): {r_ep:.2f}, epsilon: {agent.epsilon:.4f}")
    # save q-table
    os.makedirs('results', exist_ok=True)
    agent.save('results/q_table.npy')
    # plot rewards
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward (sum of step rewards)')
    plt.title('Training Rewards')
    plt.grid(True)
    plt.savefig('results/training_plot.png')
    plt.close()
    return agent, env, rewards

if __name__ == '__main__':
    train()
