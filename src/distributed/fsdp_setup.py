# PyTorch FSDP (Fully Sharded Data Parallel) configuration

# Sharding strategy determines what gets sharded across GPUs.
# FULL_SHARD: shards parameters, gradients, AND optimizer states across all ranks.
#   Maximum memory savings; use this for large models (e.g. 30B-70B) or when
#   a single GPU cannot hold the model plus optimizer states/activations.
#   Incurs extra communication (all-gather before compute, reduce-scatter after)
#   compared to lighter strategies.
# SHARD_GRAD_OP: shards only gradients and optimizer states, keeps full parameters
#   replicated on each GPU (similar to ZeRO Stage 2). Use this when the model
#   itself fits comfortably in GPU memory but optimizer states don't (smaller
#   models, e.g. up to ~7-8B), since it reduces communication overhead vs FULL_SHARD
#   and trains faster.
sharding_strategy: "FULL_SHARD"

# Auto-wrap policy: automatically wraps individual transformer blocks as their
# own FSDP units (rather than the whole model as one unit), which is what enables
# fine-grained sharding/prefetching. Name the actual decoder layer class from the
# model implementation, e.g. LlamaDecoderLayer for Llama-family models.
auto_wrap_policy: "transformer_auto_wrap_policy"
transformer_layer_cls_to_wrap:
  - "LlamaDecoderLayer"

# Controls when the next FSDP unit's parameters are prefetched during the
# backward pass. BACKWARD_PRE prefetches earlier (before current unit's backward
# compute finishes), overlapping communication with compute more aggressively;
# BACKWARD_POST prefetches after, using less peak memory but less overlap.
backward_prefetch: "BACKWARD_PRE"

# Whether to offload parameters/gradients/optimizer states to CPU when not in use.
# false = keep everything on GPU (fastest, requires more GPU memory).
# Set to true only if GPU memory is insufficient even with FULL_SHARD, since
# CPU offload adds significant PCIe transfer overhead and slows training.
cpu_offload: false

# Mixed precision dtype used for parameters, gradient reduction, and buffers
# during FSDP's internal compute/communication. bf16 is recommended on Ampere+
# GPUs for training stability.
mixed_precision:
  param_dtype: "bf16"
  reduce_dtype: "bf16"
  buffer_dtype: "bf16"

# Enables activation checkpointing (recompute activations in the backward pass
# instead of storing them), trading extra compute for significantly lower
# activation memory. Recommended for large models under FSDP.
activation_checkpointing: true