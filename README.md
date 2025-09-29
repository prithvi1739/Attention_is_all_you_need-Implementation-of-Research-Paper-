# Attention_is_all_you_need-Implementation-of-Research-Paper-
This repo is my clean, from-scratch implementation of the Transformer architecture introduced in Attention Is All You Need (Vaswani et al., 2017). The core idea is simple: stop forcing the model to read tokens one after another with a recurrent loop. Instead, let it look at everything at once and decide what matters using attention.

Why attention matters

Here’s the thing: in language, the useful signal for a token might sit one step away or fifty. RNNs and LSTMs have to carry that signal through a long chain, which leads to vanishing gradients, memory bottlenecks, and slow training. Self-attention skips the line:

It directly connects every token to every other token in a single step.

It learns which tokens to focus on via query-key-value scoring.

It’s position-aware (via positional encodings) without dragging state through time.

It scales naturally to long-range dependencies and multi-token patterns.

In practice this means better gradient flow, clearer inductive bias for relationships, and stronger performance with less pain.

From sequential recurrence to parallelization

Traditional sequence models (RNN/LSTM/GRU) process tokens in order: step t depends on step t-1. That recurrence blocks parallelism. The Transformer removes this bottleneck:

No recurrence: Self-attention computes interactions across the full sequence in one shot.

Full parallel training: All time steps are processed simultaneously on the GPU/TPU.

Depth over time: We trade “time steps” for layers, stacking attention + feedforward blocks.

What this really means is you get orders-of-magnitude faster training and the freedom to scale width and depth without waiting for a for-loop over timesteps.

What’s in this repo

Minimal, readable PyTorch modules for:

Multi-Head Self-Attention (scaled dot-product)

Position-wise Feedforward layers

Residual connections + LayerNorm

Positional encodings (sinusoidal)

Encoder/Decoder stacks with masked attention for autoregressive decoding.

Roadmap

This is just the beginning. I’ll be implementing more research papers across ML and Generative AI—variants of attention, modern pretraining tricks, and domain-specific adaptations. Expect a steady stream of clear, production-ready reproductions with strong baselines and concise notes.

Reference

Vaswani et al., 2017. Attention Is All You Need. NIPS.

If you spot something off or want a feature, open an issue. Let’s make it sharper together.

Training script with teacher forcing, label smoothing, and warmup learning rate schedule

Tiny config system for quick ablations (heads, depth, model dim)
