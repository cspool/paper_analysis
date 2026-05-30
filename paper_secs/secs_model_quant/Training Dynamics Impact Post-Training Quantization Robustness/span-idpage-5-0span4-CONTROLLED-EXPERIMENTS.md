# <span id="page-5-0"></span>4 CONTROLLED EXPERIMENTS

#### <span id="page-5-2"></span>4.1 REPLICATING THE OBSERVED PHENOMENA

To understand the insights from Section 3, we conduct pretraining experiments with transformer models on a smaller scale, varying token budget, learning rate, LR schedule, and weight decay one at a time. We follow Biderman et al. (2023) for model specifications, and FineWebEdu (Penedo et al., 2024) as pretraining corpus (see Appendix C for details on the training procedure and hyperparameters). We use GPTQ, and discuss results for additional quantization methods in Appendix A.

In Figure 4 we show quantization error and validation loss across a range of token budgets, which we obtain by decaying the learning rate at different steps during training. We observe that the constant learning rate stage is not immune to PTQ degradation, showing a slight increase in quantization error. At the same time, despite training durations ranging from 10B to 100B tokens, models achieve *comparable quantization error* after decay, which spikes as learning rate decays and validation loss drops, regardless of than token count. In Figure 21 we replicate the experiment using a cosine decay schedule, where model performance (Figure 21b) and quantization robustness (Figure 21a) vary with the training horizon. However, changes in the peak learning rate, and thus the scheduler shape, have a larger impact, in some cases yielding improved quantization error at lower validation loss.

In conclusion, this evidence suggests that the phenomena observed in Section 3 are not merely serendipitous outcomes of complex model interactions, but are strongly shaped by training dynamics, with factors such as learning rate decay playing a key role in quantization performance.

#### 4.2 SCALING TRENDS IN PRIOR WORK ARE DOMINATED BY LEARNING RATE SCHEDULES

In an effort to explain the rise of quantization error during training, previous studies attributed this phenomenon to dataset size or training duration, concluding that *PTQ degradation increases as models are trained on more data* (Kumar et al., 2024), and hence that quantized undertrained models scale more favorably (Ouyang et al., 2024). We argue that these works did not sufficiently control for a key confounder, namely the optimization dynamics induced by the learning rate schedule, which we find to be the primary driver of their observed degradation.

Specifically, we replicate analyses from Kumar et al. (2024) in Figure 5, training models at different token budgets under both original cosine schedule and WSD schedule. While cosine results (blue) suggest that  $\delta_{PTQ}$  increases noticeably with token budget, we show that a comparable WSD schedule (brown) can yield lower validation loss, with degradation growing more slowly (70M) or remaining stable (160M), indicating that the effect cannot be ascribed to data alone (see also Figure 21 for a similar conclusion).

Finally, we argue for additional caution when collecting checkpoints at different token counts, as done in Ouyang et al. (2024). We recall that similar considerations have been discussed in the scaling law literature: Hoffmann et al. (2022) suggested that their power law discrepancy with Kaplan et al.

<span id="page-6-1"></span>![](_page_6_Figure_1.jpeg)

Figure 5: Learning rate affects quantization scaling trends. Following Kumar et al. (2024), we train 70M and 160M transformer models with cosine decay across different token budgets, and a WSD schedule under the same model configurations. Cosine decay replicates prior results, with  $\delta_{PTQ}$  increasing at larger token budgets, while WSD shows slower growth at 70M and no increase at 160M, hinting that other factors beyond data volume influence quantization scaling.

(2020) arose from differences in learning rate schedules, and further works validate the importance of collecting checkpoints only after learning rate annealing (Haegele et al., 2024). We suggest that the same discretion is necessary when deriving scaling laws for quantized models, as optimization dynamics influence observed robustness (Figure 4).

