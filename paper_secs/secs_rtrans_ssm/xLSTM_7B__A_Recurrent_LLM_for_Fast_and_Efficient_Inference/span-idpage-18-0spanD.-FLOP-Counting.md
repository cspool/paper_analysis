# <span id="page-18-0"></span>**D. FLOP Counting**

We count the number of FLOPs in a forward pass of the mLSTM. We use a factor of 2 to describe the multiply accumulate cost.

We use factors denoted as  $F_X$  to describe the number of FLOPs for operation X (e.g.  $F_exp$  for the exponential function). By default we set all of these factors to 1.

### D.1. FLOPs for the mLSTM Operation

- Inter-chunk recurrent:
  - Chunkwise gates: num\_heads × num\_chunks
    - $\times$  (0.5×chunk\_size  $\times$  (chunk\_size + 1) + 2×chunk\_size)
  - Gates & max state: num\_heads × num\_chunks
    - $\times$  (3 + F\_max + F\_exp + chunk\_size  $\times$  (3 + 2  $\times$  F\_exp))
  - **Numerator:** num\_heads × num\_chunks
    - $\times (2 \times d_q k \times d_v + 4 \times chunk\_size \times d_q k \times d_v + 3 \times chunk\_size \times d_q k)$
  - **Denominator:** num\_heads  $\times$  num\_chunks  $\times$  ( d\_qk + 4 $\times$ chunk\_size  $\times$  d\_qk )
- Intra-chunk parallel:
  - Gate matrix: num\_heads × num\_chunks
    - $\times$  (0.5  $\times$  chunk\_size  $\times$  (chunk\_size + 1)
    - + chunk\_size  $\times$  chunk\_size  $\times$  (3 + F\_mask + F\_max + F\_exp)
    - + chunk\_size  $\times$  (1 + F\_max))
  - Gated Attn logits: num\_heads × num\_chunks
    - $\times 2 \times \text{chunk\_size} \times \text{chunk\_size} \times (1 + d_{qk})$
  - **Numerator:** num\_heads × num\_chunks
    - $\times$  2×chunk\_size × chunk\_size × d\_v

- Denominator: num heads × num chunks × 2 × chunk size × chunk size
- Output combination: num heads × num chunks × ( chunk size × ( 1 + F max )
  - + chunk size × ( 2 + F abs + F exp + F max + 2×d v ) )

### D.2. FLOPs for the mLSTM in a Transformer Backbone

For computing the number of FLOPs we follow the procedure from [Hoffmann et al.](#page-10-18) [\(2022\)](#page-10-18). We include the FLOPs contributed by the embedding matrices. We do not include RMS- or Layer-Norm and skip connection FLOPs We assume that the backward pass has 2 times the number of FLOPs of the forward pass. For the forward pass, the number of FLOPs of the mLSTM for a single sequence can be approximated by:

- Embeddings
  - 2 × seq len × vocab size × d model
- mLSTM (single layer)
  - Query, key, value, input and forget gate projections:
    - 2 × seq len × d model × num heads × (2 × d qk + d v + 2)
  - Output gate and projection:
    - 4 × seq len × d model × num heads × d v
    - + seq len × num heads × d v × F sig
  - mLSTM cell: See above.
- Gated Feedforward (single layer)
  - 6 × seq len × d model × d model × proj factor ff + 2 × seq len × d model × F swish
- Final Logits
  - 2 × seq len × d model × vocab size
- Total forward pass FLOPs:

embeddings + num layers × (mLSTM + feedforward) + final logits

## D.3. FLOPs for the Transformer with Self-Attention

We use the FLOP computations from [Hoffmann et al.](#page-10-18) [\(2022\)](#page-10-18), with the difference that we use gated feedforward blocks.

- Embeddings
  - 2 × seq len × vocab size × d model
- Attention (single layer)
  - Key, query and value projections:

```
2 × seq len × d model × num heads × (2 × d qk + d v)
```

- Key @ query logits: 2 × seq len × seq len × (d qk × num heads)
- Softmax: 3 × seq len × seq len × num heads
- Softmax @ query reductions: 2 × seq len × seq len × (num heads × d qk)
- Final linear: 2 × seq len × d model × (num heads × d v)
- Gated Feedforward (single layer)
  - 6 × seq len × d model × d model × proj factor ff
    - + 2 × seq len × d model × F swish

- Final Logits
  - 2 × seq len × d model × vocab size
- Total forward pass FLOPs:

embeddings + num layers × (attention + feedforward) + final logits

