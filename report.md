# Report: Traffic Light Optimization using Reinforcement Learning (Simplified)

## Objective
Use Reinforcement Learning (Q-learning) to learn signal timing policies that reduce waiting at a single intersection.

## Implementation Summary
- Environment (`environment.py`): Simulates arrivals and departures at an intersection with two phases.
- Agent (`agent.py`): Q-learning agent with a discrete Q-table.
- Training (`train.py`): Trains agent for multiple episodes and saves Q-table and reward plot.
- Main (`main.py`): Runs evaluation before and after training and saves comparison plot.

## Reward Design
Reward is negative of total queue length at each step:
`reward = - (ns_queue + ew_queue)`

## How to Run
See README.md

## Notes and Possible Improvements
- Use SUMO for realistic traffic flow.
- Replace Q-table with DQN for larger state spaces.
- Add multiple intersections and communication between agents.
