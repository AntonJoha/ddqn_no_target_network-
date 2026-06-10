# ddqn_no_target_network-

Standard Double DQN (DDQN) implementation for LunarLander is provided in:

- `ddqn_lunar_lander.py`

## Install dependencies

```bash
pip install gymnasium[box2d] torch numpy
```

## Run training + evaluation

```bash
python ddqn_lunar_lander.py --episodes 500
```

Useful options:

- `--env-id LunarLander-v3`
- `--seed 42` (use a different seed for each run; saved files include the seed)
- `--render` (renders environment while training/evaluating)

## Notebook cleanup

```bash
python clear_notebooks.py
```