# E. Parameter Counting

In this section we count the number of paramters in the mLSTM and compare it to the number of parameters in a Transformer with self-attention. We assume that the model does not use weight tying and omits biases.

### E.1. Parameter Counting for the mLSTM

- Embeddings
  - vocab size × d model
- mLSTM (single layer)
  - qkv: d model × num heads × (2 × d qk + d v)
  - Input and forget gate: 2 × d model × num heads + 2 × num heads
  - Output gate: d model × d model
  - Output projection: d model × d model
  - Norm: d model
- Gated Feedforward (single layer)
  - 3 × d model × d model × proj factor ff
- Norm (single layer)
  - d model
- Final Logits:
  - d model × vocab size
- Total number of parameters:

embeddings + num layers × (mLSTM + feedforward + 2 × norm) + norm + final logits

### E.2. Parameter Counting for the Transformer with Self-Attention

- Embeddings
  - vocab size × d model
- Attention (single layer)
  - qkv: d model × num heads × (2 × d qk + d v)
  - Output projection: d model × d model
- Gated Feedforward (single layer)
  - 3 × d model × d model × proj factor ff
- Norm (single layer)
  - d model
- Final Logits:
  - d model × vocab size

## • Total number of parameters:

embeddings + num layers × (attention + feedforward + 2 × norm) + norm + final logits