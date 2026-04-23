# ACT v5 Checkpoint

Trained on v5 dataset (30 HD demos with `call_button_lit` observation).

## Files
- `dataset_stats.pkl` — qpos/action normalization stats (mean, std) used at inference.
- `train.log` — full training log from epoch 0 → 400 (val loss 84 → 0.20).
- `policy_epoch_400_seed_0.ckpt` — trained ACT state_dict, **uploaded as a GitHub Release asset**
  (too large for the git tree).

## Hyperparameters
- Policy class: ACT
- chunk_size: 50
- kl_weight: 10
- hidden_dim: 512
- dim_feedforward: 3200
- batch_size: 8
- learning rate: 1e-5
- seed: 0
- state_dim: 8 (7 joint + 1 LED)

## Training signal
- Rapid decay: 0 → 40 epochs (val loss 84 → 0.6)
- Slow convergence: 40 → 400 (val loss 0.6 → 0.2)
- Oscillation plateau around 0.22 after epoch 200.
