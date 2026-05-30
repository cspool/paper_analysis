# A.1 Training Performance under Ideal Failure Patterns for Tutel and Tutel(FT)

In Figure 14 we present the training performance under a single node failure every 5 minutes. We ensure that subsequent failed nodes are nodes that are previously dropped by Tutel and Tutel(FT), as the total number of nodes is not a multiple of EP size. We keep other settings the same as in §6.6.

In this case, after initial failures at 5 minutes, Tutel and Tutel(FT) essentially only encounter a failure every 10 minutes for GPT-S and GPT-M, due to an EP size of 2; every 20 minutes for GPT-L, due to an EP size of 4. In terms of total trained samples, Lazarus outperforms Tutel(FT) by 1.4x for GPT-S and by 2.6x for GPT-L.

