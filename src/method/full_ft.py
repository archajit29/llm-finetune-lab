# Method configuration for full-parameter fine-tuning with HuggingFace Trainer
# No adapter/PEFT fields — all model parameters are updated during training.

# Peak learning rate for the optimizer.
# Typical range for FULL fine-tuning: 1e-5 to 5e-5 (small, since all weights update
#   and large updates can easily destabilize/overfit a fully-tuned model).
# For comparison, LoRA typically uses a much higher LR, ~1e-4 to 3e-4, since only
#   small adapter matrices are trained and can tolerate/require larger steps.
learning_rate: 2.0e-5

# Fraction of total training steps used to linearly warm up the LR from 0 to
# the peak learning_rate, before decaying. Helps stabilize early training.
warmup_ratio: 0.03

# L2 regularization coefficient applied to weights (excluding biases/LayerNorm
# by default in most Trainer setups). Helps prevent overfitting during full FT.
weight_decay: 0.01

# Enables gradient checkpointing: trades compute for memory by not storing all
# intermediate activations, recomputing them during the backward pass instead.
# Essential for full fine-tuning of large models due to high memory demands.
gradient_checkpointing: true

# Optimizer implementation. adamw_torch is the standard PyTorch AdamW.
# adamw_bnb_8bit (bitsandbytes 8-bit AdamW) cuts optimizer state memory ~4x,
# useful for full FT on memory-constrained GPUs, with minimal quality impact.
optim: "adamw_torch"

# Maximum gradient norm for gradient clipping, used to prevent exploding
# gradients and stabilize training, especially early in full fine-tuning.
max_grad_norm: 1.0