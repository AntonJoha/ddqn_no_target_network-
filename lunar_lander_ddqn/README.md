# Lunar Lander DDQN

This folder contains a LunarLander version of the DDQN example.

## Install

```bash
pip install gymnasium[box2d] torch numpy
```

## Run

```bash
python lunar_lander_ddqn/ddqn_lunar_lander.py --episodes 500
```

Use `--seed` to keep separate runs from overwriting each other's saved models and outputs.
