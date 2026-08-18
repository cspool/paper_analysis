# III. COSET ENSEMBLE DECODER

This section presents three algorithmic contributions that together define our coset ensemble decoding procedure (Algorithm [1\)](#page-3-1): *ensemble forest exploration* (Sec. [III-A\)](#page-2-1), *reverseorder elimination* (Sec. [III-B\)](#page-4-0), and *lossless graph compression* (Sec. [III-C\)](#page-5-0).

#### <span id="page-2-1"></span>*A. Ensemble Forest Exploration*

*1) Algorithm Overview:* We adopt the coset viewpoint of stabilizer decoding: any error E can be decomposed as E = s(E)t(s)l(E), where the syndrome s fixes t(s) and ambiguity

![](_page_3_Figure_0.jpeg)

<span id="page-3-0"></span>Fig. 3. Latency breakdown for code distances 3 (left) and 11 (right) on a twostage baseline without optimizations. Clustering Pipeline Stall and Clustering Pipeline Busy are cycles spent inside the *pipelined clustering engine* (stalled vs. productive); Spanning Tree and Peeling are cycles spent in the *postclustering traversal modules*.

# III. COSET ENSEMBLE DECODER

This section presents three algorithmic contributions that together define our coset ensemble decoding procedure (Algorithm [1\)](#page-3-1): *ensemble forest exploration* (Sec. [III-A\)](#page-2-1), *reverseorder elimination* (Sec. [III-B\)](#page-4-0), and *lossless graph compression* (Sec. [III-C\)](#page-5-0).

#### <span id="page-2-1"></span>*A. Ensemble Forest Exploration*

*1) Algorithm Overview:* We adopt the coset viewpoint of stabilizer decoding: any error E can be decomposed as E = s(E)t(s)l(E), where the syndrome s fixes t(s) and ambiguity

![](_page_3_Figure_0.jpeg)

<span id="page-3-0"></span>Fig. 3. Latency breakdown for code distances 3 (left) and 11 (right) on a twostage baseline without optimizations. Clustering Pipeline Stall and Clustering Pipeline Busy are cycles spent inside the *pipelined clustering engine* (stalled vs. productive); Spanning Tree and Peeling are cycles spent in the *postclustering traversal modules*.

