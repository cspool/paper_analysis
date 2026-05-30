# C.1 Detailed Explanation of FreqFold and RoRoPE's PCA

In the RoRoPE framework, Rotary Position Embedding (RoPE) is applied. RoPE encodes positional information by rotating pairs of feature dimensions. For each RoPE frequency index  $l \in \{1,\ldots,d/2\}$ , the corresponding pair of dimensions ([2l-1::d],[2l::d]) from query and key vectors are rotated. When multiple original attention heads are used (say, g heads), and their key/query projection outputs are concatenated, the RoPE operation for a specific frequency index l applies to a 2g-dimensional vector segment (formed by concatenating the l-th 2D RoPE subspace from each of the g heads). RoRoPE then applies PCA via matrices  $\{\mathbf{U}_l\}_{l=1}^{d/2}$  to these 2g-dimensional segments, independently for each frequency index l.

The core idea of FreqFold is to approximate numerically similar RoPE base frequencies as being effectively identical. For instance, if RoPE uses original base frequencies  $\theta_{l_1}, \theta_{l_2}, \dots, \theta_{l_M}$  that are close in value, MD-FreqFold might treat them all as a single, representative frequency  $\theta^*$ .

This approximation has a significant implication for how PCA is applied in RoRoPE:

- Without FreqFold (Standard RoRoPE PCA): For each distinct RoPE frequency index l, a separate PCA transformation U<sub>l</sub> is learned and applied to the corresponding 2g-dimensional key/query segments.
- With FreqFold: If M original RoPE frequency indices (say  $l_1, \ldots, l_M$ ) are grouped together by FreqFold due to their frequency similarity, the M corresponding 2g-dimensional segments are effectively concatenated. Instead of M separate PCAs on 2g-dimensional vectors, a single PCA is performed on the resulting  $M \cdot 2g$ -dimensional vectors.

