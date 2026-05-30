# <span id="page-13-0"></span>5 Limitations and Future work

Due to the top-k selection over tokens, MoSA is non-autoregressive in nature and requires adaptations to be directly applicable to the autoregressive scenario. This is true not only for MoSA, but for all expert-choice routing methods, as well as for the Routing Transformer that uses non-autoregressive clustering. MoD proposed to solve this problem by learning an autoregressive classifier post-training to predict if the given token would have been selected by the non-autoregressive router or not. We consider exploring this issue in depth as an important future direction.

The perplexity gains do not always translate to downstream task performance (Section [3.5\)](#page-10-1). This discrepancy stems from two distinct factors: First, sparse attention methods generally underperform on tasks consisting of short sequence lengths. Practitioners have shown that additional training with truncated sequences might alleviate this problem. Second, MoE architectures experience performance gaps in downstream tasks despite strong language modeling capabilities, although recent research demonstrates that instruction tuning can help significantly [\[58\]](#page-17-5). We consider exploring methods to mitigate the discrepancy between perplexity and downstream task performance in future work.

Several promising research directions emerge from this work. Further exploration of MoSA's effectiveness on longer sequences remains an important direction. Furthermore, combining multiple sparse attention methods often leads to synergic improvements on long sequences [\[19,](#page-15-0) [20\]](#page-15-1). Thus, we expect that combining other sparse head types with MoSA could lead to additional benefits.

From an implementation perspective, developing specialized CUDA kernels would further improve efficiency. MoSA could be integrated with complementary approaches such as MQA[\[36\]](#page-16-0), GQA[\[69\]](#page-17-16), or SwitchHead[\[31\]](#page-15-12) to improve the efficiency even further.

Furthermore, exploring MoSA on other modalities, particularly vision transformers, could yield valuable insights into the method's versatility across different data types and architectures.

