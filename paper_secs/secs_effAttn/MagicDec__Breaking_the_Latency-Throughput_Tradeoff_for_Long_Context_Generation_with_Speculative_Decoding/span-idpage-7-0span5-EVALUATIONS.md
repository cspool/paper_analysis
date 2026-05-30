# <span id="page-7-0"></span>5 EVALUATIONS

In this section, we empirically validate our theoretical analysis and demonstrate the effectiveness of our drafting strategy selection modeling. Specifically, in Section [5.1,](#page-7-2) we demonstrate the end-to-end speedup of self-speculation with sparse KV, showing that speculative decoding achieves speedup for moderate-to-long sequences, with speedup increasing as batch size grows, when sequence length exceeds a critical threshold. In Section [5.2,](#page-7-3) we compare the speedup of two drafting strategies, highlighting the effectiveness of our approach. In Section [5.3,](#page-8-0) we perform an ablation study on the speedup of speculative decoding.

