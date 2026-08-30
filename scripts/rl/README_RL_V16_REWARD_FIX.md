# OpenArm RL V16 Reward Fix

V15 reached fixed-cube 10/10 but random-cube only around 3/30. The main issue was reward hacking: repeated per-step lift bonuses allowed large reward without final placement.

V16 changes:

- one-time lift bonus instead of repeated per-step bonus
- one-time near-frame bonus
- bounded progress rewards
- timeout penalty
- shorter default max_steps=450
- more gradual curriculum: tiny -> near -> medium -> full

Run:

```bat
python scripts\rl\train_right_pick_place_curriculum_v16.py
python scripts\rl\test_right_pick_place_policy_compact_v16.py --episodes 30
```

If training is too long, reduce the last stage:

```bat
python scripts\rl\train_right_pick_place_curriculum_v16.py --stage4 200000
```
