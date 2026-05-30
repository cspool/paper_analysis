# V. IMPLEMENTATION

We implement our system in PyTorch [70] (v2.0). We encapsulate key components such as *adaptive all-to-all*, *dynamic expert clustering*, and *topology-aware expert remapping*, making it highly applicable to various distributed training frameworks (e.g., DeepSpeed, Megatron-LM). For the prototype, we build ScaleMoE on top of DeepSpeed, one of the state-of-theart distributed training frameworks. Through extensive stress tests in diverse training setups, we confirm that ScaleMoE is robust enough to support various real-world scenarios.

The *dynamic expert clustering* and *topology-aware expert remapping* require some CPU execution time; therefore, we need to minimize such overheads. We divide an epoch into

![](_page_7_Figure_9.jpeg)

(a) Timeline w/o overlapping.

(b) Timeline w/ overlapping.

Fig. 12: Execution timeline comparison between without overlapping (a) and with overlapping (b).

smaller units (called a *superbatch*) and perform clustering and remapping for each superbatch. Since each superbatch is independent, we can overlap these operations with GPU iterations. Figure 12 illustrates how our overlapping technique can eliminate the clustering-related operations (i.e., clustering & remapping, clustering for input). As shown in Figure 12a, without overlap, the clustering-related operations (on the CPU) and main iterations (on the GPU) are executed sequentially. In this case, all CPU-side operations take 7.79 seconds, increasing overall execution time by 12.48% compared to the GPU-only scenario. On the other hand, as shown in Figure 12b, overlapping ensures that all CPU-side clusteringrelated operations are executed concurrently with main iterations. Therefore, it can eliminate clustering-related overheads, reducing them to just 0.001% of the GPU iteration time. Note that using a superbatch does not negatively impact clustering efficiency. We conduct in-depth sensitivity analyses across different superbatch sizes in Section VI-D.

We emphasize that ScaleMoE maintains the integrity of the original training process. First, the replicated experts are correctly updated after the backward pass of each iteration ensuring consistency and computational correctness. Second, to preserve the sequence and position information of individual tokens, we transmit <*sequenceID*, *tokenIndex*, *tokenName*> along with the tokens. By doing so, we ensure that positional data for individual tokens remain intact throughout the training process. Lastly, before the output layer, tokens are reordered to their original sequence. This reordering step involves an additional all-to-all operation; however, the overhead is minimal compared to the existing all-to-all operations. Through the evaluations, we confirm that ScaleMoE keeps the integrity of the original training process without compromising accuracy.

