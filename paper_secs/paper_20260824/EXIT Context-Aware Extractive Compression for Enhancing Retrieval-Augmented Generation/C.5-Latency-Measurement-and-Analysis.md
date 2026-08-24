# C.5 Latency Measurement and Analysis

To ensure the reliability and reproducibility of our latency results, we conducted five independent runs for each compression method and report both the mean and standard deviation of end-to-end latency (measured in seconds per sample). End-to-end latency includes both compression time and reading/inference time, measured from the moment a request is issued to the moment the answer is returned by the model.

For consistency, all experiments using the 8B reader were run on a single A100-80GB SXM GPU, while those using the 70B reader were executed on four A100-80GB SXM GPUs. To minimize hardware-induced noise, we ensured that no unrelated processes were active during measurement and maintained fixed hyperparameters and model configurations across all runs.

As shown in Table [12,](#page-19-0) our method EXIT achieves a strong balance between latency and accuracy. While abstractive methods such as CompAct and Refiner incur significant latency (12.86s and 6.78s ,respectively, with the 8B reader), EXIT completes inference in under 1 second (0.88s) on

average. Even with the 70B reader, EXIT remains efficient with 3.86s total latency, significantly faster than the original uncompressed baseline (8.08s), while also improving QA performance.

These results reinforce EXIT's practicality for real-world deployment, particularly in latencysensitive applications.

#### C.6 Impact of Threshold τ

We analyze EXIT's sensitivity to the relevance threshold τ . Figure [7](#page-19-1) shows EXIT's performance across various τ values.

EXIT remains stable over a wide threshold range, with strong results between τ=0.3–0.5. At τ=0.3, EXIT reaches 25.2 EM using only 25% of the tokens (195.82 vs. 780.95 for the baseline), a substantial improvement over the original documents (18.0 EM). F1 scores also remain consistently higher than the baseline (30.21–30.73 vs. 25.74).

Even under extreme compression (τ=0.9, 7.9% of tokens), EXIT achieves better accuracy (24.0 EM, 29.44 F1) than the uncompressed documents. Conversely, a lenient threshold (τ=0.1) retains more tokens but still provides benefits, demonstrating that EXIT effectively identifies crucial content under varying conditions.

This robustness across thresholds gives practitioners flexibility to adjust the compressionaccuracy trade-off without severely impacting performance.

#### C.7 Effect of Classifier Training

To evaluate the impact of fine-tuning the classifier in EXIT, we compared our trained model with a frozen zero-shot version using Gemma-2B (without any task-specific fine-tuning). We tested both variants on an in-domain dataset (HQA) and an out-of-domain dataset (2WIKI).

As shown in Table [13,](#page-20-1) the zero-shot classifier already improves over the uncompressed baseline, indicating the strength of relevance-based filtering even without training. However, our fine-tuned classifier consistently outperforms the zero-shot variant in both EM and F1 scores while achieving significantly greater token reduction—demonstrating that supervised training yields both better accuracy and more efficient compression.

