# PPO Notes

This implementation uses an Actor-Critic model and the PPO clipped surrogate objective:

- Policy loss: `min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)`
- Value loss: MSE between predicted value and return
- Entropy bonus encourages exploration

Advantages are computed with GAE(λ), then normalized for stability.
