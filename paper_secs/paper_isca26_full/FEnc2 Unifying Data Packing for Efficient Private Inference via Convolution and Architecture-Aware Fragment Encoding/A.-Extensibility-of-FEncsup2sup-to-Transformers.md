# A. Extensibility of FEnc<sup>2</sup> to Transformers

The core principle of our FEnc<sup>2</sup> extends naturally beyond CNNs to Transformer-based models. In our CNN design, we reduce intra-ciphertext data dependence by splitting feature maps across multiple ciphertexts, thereby decreasing the

<span id="page-11-2"></span>![](_page_11_Figure_8.jpeg)

Fig. 9: Scalability comparison against *Orion* in terms of higher-resolution input and larger CNN model.

<span id="page-11-0"></span>TABLE X: Performance of Arch-aware Ct Compression

| Layer $(N_{in}, N_{DS}, N_{out})$ | with<br>latency(s) | out AAC<br>slot utilization | latency(s) | th AAC<br>slot utilization | Latency<br>Speedup (×) |
|-----------------------------------|--------------------|-----------------------------|------------|----------------------------|------------------------|
| Fire-module (64,32,128)           | 8.84               | 0.5                         | 6.00       | 1                          | 1.47                   |
| Fire-module<br>(128,32,128)       | 12.75              | 0.25                        | 6.65       | 1                          | 1.92                   |
| Fire-module<br>(128,32,256)       | 8.69               | 0.25                        | 4.64       | 1                          | 1.95                   |
| Fire-module<br>(256,32,256)       | 10.40              | 0.063                       | 2.22       | 0.5                        | 4.68                   |
| Total                             |                    | 40.67                       | 19.51      |                            | 2.09                   |
| Residual-shortcut<br>(64,8,128)   | 2.35               | 1                           | 2.38       | 1                          | 1.016                  |
| Residual-shortcut<br>(128,8,256)  | 11.87              | 0.25                        | 6.92       | 1                          | 1.72                   |
| Residual-shortcut<br>(256,8,512)  | 4.12               | 0.06                        | 2.35       | 1                          | 1.75                   |
| Residual-shortcut<br>(512,8,512)  | 32.17              | 0.02                        | 19.21      | 1                          | 1.67                   |
| Total                             |                    | 50.50                       |            | 30.87                      | 1.64                   |

amount of costly rotation needed within each ciphertext. We apply the same idea to Transformers by splitting each token embedding across ciphertexts. Specifically, given n input tokens with embedding dimension E, we partition each embedding into sub-blocks of size S, producing E/S fragments. The input tensor is thus reorganized from  $(n \times E)$  into  $(n \times \frac{E}{S} \times S)$ , and fragments from different tokens are interleaved across ciphertexts. This fragmented layout reduces embedding-wise intra-ciphertext dependence in the same way that our CNN layout reduces feature-map-wise dependence. As a result, the number of rotations required for embedding aggregation is reduced from E-1 in full-embedding packing to  $\frac{E}{S}-1$  under fragmented encoding. Importantly, this extension preserves the same trade-off as in the CNN case. The block size Scontrols the trade-off between inner-rotation and outer-rotation overhead: a smaller S retains more computation within each ciphertext and requires more inner rotations, while a larger S reduces inner-rotation cost at the expense of increased outer-rotation overhead across ciphertexts. This trade-off also appears in Transformer components such as Feed-Forward Network (FFN) and Multi-Head Attention (MHA), where embedding-wise and token-wise rotations are similarly controlled by S.

While fragmentation optimizes FFN and projection layers, it introduces a unique trade-off during the MHA stage. Because S times more tokens are packed into each ciphertext to maintain high density, token-wise aggregation (required for  $QK^\intercal$  and AV computations) incurs a proportional increase in rotations. The total rotation complexity across both FFN and MHA layers can be formulated as  $\mathcal{O}\left(\frac{\gamma E}{S} + S \cdot 2n\right)$ , where  $\gamma$  is the magnification factor of the FFN layer relative to the embedding size. This reveals a critical optimization space for selecting the block size S, analogous to the design-space exploration required for CNNs.

We implement and evaluate the inference latency corresponding to one encoder block (including MHA and FFN) in the BERT-base model ( $E=768, n=128, \gamma\approx 6$ ). Specifically, to perform the  $QK^{\mathsf{T}}V$  computation, we modify the general HE-matrix transposition procedure with specialized token-block rotations to align data between the MHA and FFN stage transition. We additionally implement an adapted Orion packing strategy as the baseline for comparison. We validated this on an NVIDIA A6000 GPU using encryption parameters ( $N=2^{16},\log Q=1768$ ). As shown in Table XI, the total latency reaches its minimum at S=4, which demonstrates  $6.74\times$  speedup over Orion, confirming that a balanced block size effectively minimizes the combined overhead of

<span id="page-12-0"></span>TABLE XI: Latency Comparison of Transformer Encoder Block.

| Modules                      | Orion   | FEnc2  | Speedup |
|------------------------------|---------|--------|---------|
| X × WQ,K,V                   | 87.85   | 43.35  | 2.03×   |
| A = Q × K⊺                   | 1738.17 | 124.96 | 13.91×  |
| O = A × V                    | 32.39   | 26.34  | 1.23×   |
| FFN                          | 232.71  | 115.82 | 2.01×   |
| MatMul latency/Encoder Block | 2091.12 | 310.48 | 6.74×   |

embedding-wise and token-wise rotations. Note that evaluating Transformer models end-to-end requires integrating non-linear components such as Softmax and LayerNorm. However, preserving model accuracy when supporting these components under in HE inference remains challenging and is an active research problem [\[68\]](#page-14-14). Such an effort is beyond the scope of this work.

