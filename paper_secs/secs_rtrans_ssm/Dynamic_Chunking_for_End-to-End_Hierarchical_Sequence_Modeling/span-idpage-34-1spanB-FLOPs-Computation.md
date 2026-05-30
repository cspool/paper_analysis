# <span id="page-34-1"></span>**B** FLOPs Computation

We largely follow Hoffmann et al. (2022) with two marginally updated computations: (1) add computations for Mamba-2 (Dao and Gu 2024), and (2) modify computations in MLP blocks as we use the recent Transformer++ architecture. Assuming that all query, key, and value share the same num\_heads and head\_dim, we calculate the forward pass FLOPs as follows:

- Embeddings: 2 × seq\_len × vocab\_size × d\_model
- Attention:
  - QKV **projections:**  $2 \times 3 \times \text{seq\_len} \times \text{d\_model} \times (\text{num\_heads} \times \text{head\_dim})$
  - Attention Logit Calculation: 2 × seq\_len × seq\_len × (num\_heads × head\_dim)
  - Attention Score Softmax:  $3 \times \text{num heads} \times \text{seq len} \times \text{seq len}$
  - Score @ Query:  $2 \times \text{seq len} \times \text{seq len} \times (\text{num heads} \times \text{head dim})$
  - **Output projection:** 2 × seq\_len × (num\_heads × head\_dim) × d\_model
- Mamba-2:
  - XZ **projections:**  $2 \times \text{seq\_len} \times \text{d\_model} \times (2 \times \text{expand} \times \text{d\_model})$
  - $BC\Delta t$  projections:  $2 \times \text{seq len} \times d \mod l \times (2 \times d \text{ state} + \text{num heads})$
  - SSD:  $2 \times 3 \times \text{seg len} \times (\text{expand} \times \text{d model}) \times \text{d state}$
  - **Depthwise Convolution:** 2 × seq\_len × d\_model × window\_size
  - Gating:  $5 \times \text{seq\_len} \times \text{d\_model}$
  - Output projection:  $2 \times \text{seq} \text{ len} \times \text{d} \text{ model} \times \text{d} \text{ model}$
- · Gated MLP:
  - In, Gate, Out projections:  $2 \times \text{seq\_len} \times (3 \times \text{d\_model} \times \text{ffw\_size})$
  - Gating:  $5 \times \text{seq\_len} \times \text{d\_model}$
- Logit Prediction Head: 2 × seq\_len × vocab\_size × d\_model

We assume the backward pass consumes twice the FLOPs of the forward pass.

