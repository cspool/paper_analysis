# 4 Large Scale Upcycling

To ensure our proposed recipes work on large scale (model size and token count) regimes, we compared upcycling against continued training Nemotron-4 15B base model on 1T tokens. We decided to test only 2 variants: (1) E8G8T8 and (2) E8G1T2, due to the high training compute requirement (0.3 yottaFLOPs). We chose E8G8T8 with virtual group initialization since it worked better than E8G1T1 in our ablations. We chose E8G1T2 to show the effect of increase in compute and upcycling FLOPs. We chose E8G1T2 over E8G8T16 to show that our recommendations, including weight scaling along with softmax-then-topK router, work well for non-granular cases well. Non-granular use-cases are important to study since they practically achieve a better GPU FLOP utilization than their iso-FLOP granular counterparts.

For 1T upcycling, we initialized lr to min-lr of pretraining (4.5e-4) and used a peak-lr of 3e-4. We used cosine decay and decayed the lr to 1/100-th of the pretraining min-lr as done for the original continued training of Nemotron-4 15B [\[25\]](#page-15-10). We also used a higher batch size for the E8G8T8 model than E8G1T2 since E8G1T2 receives more tokens per expert on average. As shown in Table [1,](#page-12-0) upcycled E8G8T8 (64 experts top-8 1/8 expert hidden size) achieved a 4.1% lower validation loss than the dense continued training model while being iso-FLOP. It also achieved a better MMLU score of 66.2 (vs 65.3 for dense). We observed that the percentage difference in MMLU is sensitive to the continuous training data and the difference increased along with the token count - so longer token horizons favor MoE models. With increased training FLOPs, we also upcycled E8G1T2 (8 experts top-2) which achieved 5.2% lower validation loss and an even better MMLU of 67.6.

<span id="page-12-0"></span>

| Model                                  | val loss | MMLU (5 shots) |
|----------------------------------------|----------|----------------|
| Nemotron-4 15B [13]                    | 1.623    | 59.3           |
| Nemotron-4 15B continued training [25] | 1.377    | 65.3           |
| Nemotron-4 15B upcycling E8G8T8        | 1.320    | 66.2           |
| Nemotron-4 15B upcycling E8G1T2        | 1.306    | 67.6           |

Table 1: Upcycling Nemotron-4 15B on 1T tokens

