# <span id="page-18-0"></span>A Training Data

Data Composition. As described in the main paper, our training data consists of text, image, and video modalities. For text-based reasoning, we incorporate DAPO-Math (Yu et al., 2025); for image-based reasoning, we include ViRL (Wang et al., 2025a) and ThinkLite-Hard (Wang et al., 2025c). For video QA, we draw from several sources including Video-R1 (Feng et al., 2025), TVBench (Cores et al., 2024), STI-Bench (Li et al., 2025d), and MMR-VBench (Zhu et al., 2025). To enhance temporal grounding and grounding-based QA capabilities, we additionally include Charades-STA (Gao et al., 2017), ActivityNet (Fabian et al., 2015), Time-R1 (Wang et al., 2025d), and NExT-GQA (Xiao et al., 2024). All test samples from our evaluation benchmarks are manually excluded to prevent data leakage. The resulting training pool comprises approximately 137K samples.

<span id="page-18-2"></span>

| Table 10 Training Dataset. | We include text, | image, and video | data during training, | with a total of 83K samples. |
|----------------------------|------------------|------------------|-----------------------|------------------------------|
|                            |                  |                  |                       |                              |

| Туре  | Size  | Details                                                                                                                        |
|-------|-------|--------------------------------------------------------------------------------------------------------------------------------|
| Text  | 6.4K  | DAPO-Math (Yu et al., 2025)                                                                                                    |
| Image | 27.5K | ViRL (Wang et al., 2025a), ThinkLite-Hard (Wang et al., 2025c)                                                                 |
| Video | 49.4K | Video-R1 (Feng et al., 2025), TVBench (Cores et al., 2024),<br>STI-Bench (Li et al., 2025d), MMR-VBench (Zhu et al., 2025),    |
| video | 49.4K | Charades-STA (Gao et al., 2017), ActivityNet (Fabian et al., 2015), Time-R1 (Wang et al., 2025d), NExT-GQA (Xiao et al., 2024) |

Filtering Pipeline. We further curate a smaller, higherquality subset from the initial data pool. we remove samples with invalid ground-truth (using math-verify for math problems and rule-based checks for QA problems). Next, for each remaining sample, we generate 8 responses using the base model (i.e., Qwen2.5-VL-7B-Instruct (Bai et al., 2025b)) with a high temperature. A smaller LLM (i.e., Qwen3-30B-A3B-Instruct (Yang et al., 2025a)) evaluates each response against the ground truth and assigns correct/incorrect labels. Samples for which all 8 responses are correct (too easy) or all are incorrect (too hard) are discarded, as they contribute little to GRPO-based reinforcement learning, as illustrated in Figure 5. This difficulty-based filtering is applied only to QA tasks; for temporal grounding, we retain all samples to mitigate the base model's grounding weakness. After filtering, we finally obtain 83K samples. The detailed composition is listed in Table 10.

<span id="page-18-1"></span>![](_page_18_Figure_7.jpeg)

Figure 5 Distribution of per-sample accuracy in the initial training pool, estimated by evaluating 8 diverse responses per sample. Samples with all responses correct or all incorrect are considered too easy or too hard and are excluded from QA-based data.

Effectiveness of Data Filtering. To evaluate the effective-

ness of our data filtering pipeline, we analyze the results presented in Table 11. Two key observations emerge from this analysis. **First**, training solely on text data leads to a noticeable drop in performance on

video tasks compared to the Qwen baseline, suggesting a domain shift and poor generalization. Adding image data significantly improves video QA performance, particularly on VideoMMMU, highlighting the importance of image-based math and reasoning data. However, due to the absence of temporal grounding data, performance on the Charades-STA benchmark remains low. When combining text, image, and video data, the model achieves the best overall performance under both filtered and unfiltered settings. Second, in both the text+image and text+image+video configurations, removing overly easy or difficult samples leads to consistent performance gains. Additionally, this filtering reduces the number of training samples, thereby improving training efficiency. These findings validate the effectiveness of our data filtering strategy for GRPO-based reinforcement learning.

<span id="page-19-1"></span>Table 11 Performance Comparison across Different Training Data and Filtering Strategy. Note that we report the results under the RL with CoT setting. Combining text, image, and video data yields the best overall performance. Filtering out overly easy and hard samples consistently improves results while reducing dataset size, validating the effectiveness of our data curation pipeline.

| Training Data        | Filtered | Size | VideoMME | MVBench | VideoMMMU | Charades-STA |
|----------------------|----------|------|----------|---------|-----------|--------------|
| Text                 | ✗        | 17K  | 63.3     | 62.6    | 45.8      | 38.6         |
| Image                | ✗        | 50K  | 65.6     | 66.8    | 52.8      | 40.1         |
| Video                | ✗        | 70K  | 64.7     | 71.0    | 55.1      | 59.0         |
|                      | ✗        | 67K  | 66.1     | 67.4    | 53.3      | 41.6         |
| Text + Image         | ✓        | 34K  | 67.0     | 68.5    | 56.4      | 42.0         |
| Text + Image + Video | ✗        | 138K | 65.4     | 71.0    | 55.4      | 59.7         |
|                      | ✓        | 83K  | 66.1     | 71.2    | 56.4      | 59.8         |

