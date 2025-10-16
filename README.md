# TrafficLightRL (Simplified Python Simulation)
Traffic Light Optimization using Reinforcement Learning (Q-learning)
This is a simplified, self-contained Python project that simulates a single intersection
with two phases: North-South green (NS) and East-West green (EW). An agent learns when
to switch lights to minimize total vehicle waiting time using Q-learning.

## Requirements
- Python 3.8+
- pip packages in `requirements.txt`

## Quick start
```bash
pip install -r requirements.txt
python main.py
```

Outputs:
- `results/training_plot.png` : training reward curve
- `results/avg_wait_before_after.png` : comparison before vs after training
- `results/q_table.npy` : saved Q-table after training
