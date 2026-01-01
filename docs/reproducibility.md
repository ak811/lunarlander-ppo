# Reproducibility

Reproducibility is handled by:
- Seeding Python / NumPy / PyTorch
- Seeding Gymnasium env reset and action/observation spaces when supported
- Saving full `config.yaml` per run

Note: physics engines and GPU kernels can still introduce nondeterminism.
