# Experiments

## Experimental Settings

Evaluation Benchmarks. To evaluate the effectiveness of the proposed TSPO, we conduct experiments on four widely used benchmarks in long-form video understanding.

- LongVideoBench (Wu et al. 2024). We evaluate the validation set with 1,337 videos (avg. 12min), following standard academic protocols (Zhang et al. 2024d).
- MLVU (Zhou et al. 2024). The video length ranges from 3 minutes to 2 hours. We evaluate on the "M-Avg" portion of the "Dev" split, following (Zhang et al. 2024d).
- Video-MME (w/o sub) (Fu et al. 2024) comprises 900 videos with variable durations: short (< 2 min), medium (4∼15 min), and long (30∼60min), containing 2700 QA.
- LVBench (Wang et al. 2024c) is an extremely long video benchmark, with an average video length of 4,101 seconds—4 times longer than VideoMME.

Implementation Details. The model was trained on 8 NVIDIA A800 80GB GPUs with a single epoch, using a learning rate of 5 × 10<sup>−</sup><sup>4</sup> and a batch size of 1. The temporal agent is built upon a frozen CLIP-Large model (400M parameters) and incorporates only 3.5M learnable parameters. For the Video-MLLM that guides TSPO training, we adopt LLaVA-Video (Zhang et al. 2024d) with its parameters kept frozen. The number of candidate frames (Tc) is set to 1 FPS, while the selected frame count (Ts) is set to 64 during inference and 16 during training. The window size w is set to 12, τ is set at 0.025 and annealed to 0.01. To ensure reproducibility during inference, deterministic predictions were enforced by removing the Gumbel noise.

Comparison to the State-of-the-art. As shown in Tab. 1, our LLaVA-Video-7B\*+TSPO achieves state-of-the-art performance across four general long video benchmarks. Compared to LLaVA-Video-7B\*, we improve by +5.0% on LongVideoBench, +6.0% on MLVU, +5.1% on LVBench, and 1.1% on VideoMME. Compared to Qwen2.5VL\*, we improve by +4.9% on LongVideoBench, +11.2% on MLVU, and 1.8% on VideoMME. The modest improvement on VideoMME can be attributed to its emphasis on holistic video comprehension rather than localizations on specific keyframes. Our consistent outperformance over state-of-theart methods further validates the superiority of our approach in extracting key information from long videos.

Selector Parameters. Experimental comparisons show that our lightweight selector (Temporal Agent) achieves both parameter efficiency and superior model performance. Compared to LongVU's (Shen et al. 2024) 1B-parameter DI-NOV2 (Oquab et al. 2023) selector, our method achieves

| Model                                 | Frames | LLM Size | Selector  | LongVideoBench | MLVU | Video-MME |         | LVBench |
|---------------------------------------|--------|----------|-----------|----------------|------|-----------|---------|---------|
|                                       |        |          |           | Val            | Dev  | Long      | Average | -       |
| GPT-40 (Hurst et al. 2024)            | -      | -        | Uniform   | 66.7           | 64.6 | 65.3      | 71.9    | -       |
| GPT-4V (OpenAI 2023)                  | -      | -        | Uniform   | 61.3           | 49.2 | 53.5      | 59.9    | -       |
| Gemini-1.5-Flash (Team et al. 2023)   | -      | -        | Uniform   | 61.6           | -    | 61.1      | 70.3    | -       |
| Gemini-1.5-Pro (Team et al. 2023)     | -      | -        | Uniform   | 64             | -    | 67.4      | 75.0    | 33.1    |
| Video-LLaVA (Lin et al. 2023)         | 8      | 7B       | Uniform   | -              | 36.2 | -         | 39.9    | -       |
| Oryx-1.5 (Liu et al. 2025)            | 64     | 7B       | Uniform   | 56.3           | -    | 51.2      | 58.8    | -       |
| LLaVA-Onevision (Li et al. 2024)      | 32     | 7B       | Uniform   | 56.4           | 64.7 | 46.7      | 58.2    | -       |
| NVILA (Liu et al. 2024d)              | 1024   | 7B       | Uniform   | 57.7           | 70.1 | 54.8      | 64.2    | -       |
| Apollo (Zohar et al. 2024)            | 2FPS   | 7B       | Uniform   | 58.5           | 68.7 | -         | 61.3    | -       |
| mPLUG-Owl3 (Ye et al. 2024)           | 128    | 7B       | Uniform   | 59.7           | 70.0 | 50.1      | 59.3    | 43.5    |
| LongVU (Shen et al. 2024)             | 1FPS   | 7B       | DINOv2-1B | -              | 65.4 | -         | 60.6    | -       |
| MLLM-VFS (Hu et al. 2025b)            | 32     | 7B       | MLLM-1.5B | 57.0           | -    | 51.9      | 58.7    | -       |
| LLaVA-Video-7B (Zhang et al. 2024d)   | 64     | 7B       | Uniform   | 58.2           | 70.8 | -         | 63.3    | -       |
| LLaVA-Video-7B+TPO (Li et al. 2025)   | 64     | 7B       | Uniform   | 60.1           | 71.1 | 55.4      | 65.6    | -       |
| LLaVA-Video-7B+CoS (Hu et al. 2025a)  | 64     | 7B       | MLLM-13B  | 58.9           | 71.4 | 53.8      | 64.4    | -       |
| LLaVA-Video-7B+AKS (Tang et al. 2025) | 64     | 7B       | BLIP-0.5B | 62.7           | -    | 54.0      | 65.3    | -       |
| LLaVA-Video-7B* (Zhang et al. 2024d)  | 64     | 7B       | Uniform   | 58.9           | 70.3 | 53.6      | 64.4    | 40.2    |
| LLaVA-Video-7B*+TSPO                  | 64     | 7B       | TSPO-0.4B | 63.9           | 76.3 | 54.7      | 65.5    | 45.3    |
| Qwen2.5VL* (Bai et al. 2025)          | 64     | 7B       | Uniform   | 59.0           | 65.1 | 53.3      | 63.7    | 38.3    |
| Qwen2.5VL* + TSPO                     | 64     | 7B       | TSPO-0.4B | 64.2           | 74.3 | 56.4      | 65.5    | 46.4    |

Table 1: Comparison results on four widely recognized long video understanding benchmarks, where our method achieves state-of-the-art performances with significant accuracy gain. "\*" denotes our reproduced results under 64 frames. The first three benchmarks are evaluated using lmms-eval (Zhang et al. 2024a), and LVBench is tested using its own evaluation protocol.

| Model            | Param | LongVideoBench              | MLVU                        |
|------------------|-------|-----------------------------|-----------------------------|
| LLaVA-Video      | 7B    | 58.9                        | 70.3                        |
| LLaVA-Video+TSPO | 7B    | <b>63.9</b> <sub>5.0↑</sub> | <b>76.3</b> <sub>6.0↑</sub> |
| LLaVA-Video      | 72B   | 62.4                        | 74.4                        |
| LLaVA-Video+TSPO | 72B   | <b>66.0</b> <sub>3.6↑</sub> | <b>77.3</b> <sub>2.9↑</sub> |
| Qwen2VL          | 7B    | 55.4                        | 64.0                        |
| Qwen2VL+TSPO     | 7B    | <b>59.5</b> <sub>4.1↑</sub> | <b>71.0</b> <sub>7.0↑</sub> |
| Qwen2.5VL        | 7B    | 59.0                        | 65.1                        |
| Qwen2.5VL+TSPO   | 7B    | <b>64.2</b> <sub>5.2↑</sub> | <b>74.3</b> <sub>9.2↑</sub> |

Table 2: Performance of transferring TSPO from LLaVA-Video to other Video-MLLMs without extra training, where the sampled frame number is set to **64** consistently.

a 4.9% absolute accuracy improvement on VideoMME and 10.9% on MLVU. Against Chain-of-Shot (Hu et al. 2025a)'s 13B-scale MLLM selector, our solution outperforms by 6.0% on LongVideoBench and 4.9% on MLVU.

#### **Ablation Study**

Transferring TSPO to Other Video-MLLMs. Although our method is developed based on LLaVA-Video as the Video-MLLM backbone, we explore an efficient "one-model for all" paradigm (Liu et al. 2023; Cheng et al. 2025) by transferring our learned temporal agent from LLaVA-Video-7B to other Video-MLLMs without extra training, including Qwen2VL (Wang et al. 2024a) / Qwen2.5VL-7B (Bai et al. 2025), and also extending it to LLaVA-Video-72B. As shown in Tab. 2, our method demonstrates notable

| Method      |                 | Frames | Data  | Performance     |
|-------------|-----------------|--------|-------|-----------------|
| LLaVA-Onevi | sion+FrameVOYA. | 16     | 12.5K | - /57.5         |
| LLaVA-Onevi | sion+TSPO       | 16     | 10K   | - / <b>58.7</b> |
| Qwen2VL+M   | LLM-VFS         | 32     | 1.5M  | 57.0 / 58.7     |
| Qwen2VL+TS  | SPO             | 32     | 10K   | 58.6 / 59.6     |

Table 3: Comparison with recent keyframe training methods under the same settings. The performance is evaluated on LongVideoBench and VideoMME.

| Train Data                       | $\mathbf{R}_A$ | $\mathbf{R}_T$ | Performance |
|----------------------------------|----------------|----------------|-------------|
| None                             | -              | -              | 58.9 / 64.4 |
| Comprehensive Temporal           | $\checkmark$   |                | 62.8 / 65.5 |
| Needle-in-a-Haystack             |                | $\checkmark$   | 63.4 / 64.6 |
| Needle-in-a-Haystack             | $\checkmark$   | $\checkmark$   | 63.7 / 64.9 |
| Comprehensive Temporal + Needle. |                |                | 63.8 / 65.0 |
| Comprehensive Temporal + Needle. | ✓              | ✓              | 63.9 / 65.5 |

Table 4: Ablation of data curation and reward schemes.

generalization capability: on LongVideoBench, it achieves an average 4.5% improvement; On the MLVU dataset, the average improvement is 6.3%, with Qwen2.5VL achieving a notably higher gain of 9.2%. This cross-architecture performance verifies the generalizability of our approach.

Comparison with Keyframe Training Method. Both FrameVOYAGER (Yu et al. 2024) and MLLM-VFS (Hu et al. 2025b) are recent training-based methods that require offline keyframe ranking or labeling to supervise the

| Method           | Data | E2E training | Performance |
|------------------|------|--------------|-------------|
| LLaVA-Video      | -    | ×            | 58.9 / 64.4 |
| LLaVA-Video+SFT* | 30K  | $\checkmark$ | 62.8 / 64.8 |
| LLaVA-Video+TSPO | 10K  | $\checkmark$ | 63.9 / 65.5 |

Table 5: Comparison results of SFT\* and TSPO training.

| Method             | Frames               | Token | Frame<br>Time | LLM<br>Time | Perform.    |
|--------------------|----------------------|-------|---------------|-------------|-------------|
| LLaVA-Video        | 128→64               | 13440 | 0             | 2.7         | 58.2 / 63.3 |
| LLaVA-Video + CoS  | $128 \rightarrow 64$ | 13440 | 28.4          | 2.7         | 58.9 / 64.4 |
| LLaVA-Video + TSPO | $128 \rightarrow 64$ | 13440 | 1.2           | 2.7         | 60.6 / 65.3 |
| LLaVA-Video + TSPO | $128{\rightarrow}32$ | 6720  | 1.1           | 1.3         | 59.6 / 64.8 |

Table 6: Comparison results of inference efficiency.

keyframe selector. Compared with them, our TSPO offers two advantages: 1) RL modeling: we jointly model the keyframe selection and language generation from an RL perspective, enabling end-to-end optimization with off-the-shelf video QA data, without requiring additional frame annotations like MLLM-VFS. 2) Superior sampling strategy: FrameVOYAGER requires random pre-processing sampling from frame combinations, while our approach is on-policy, dynamically sampling based on the current policy, which progressively refines the selection strategy. For a fair comparison, since neither method has been open-source, we use their reported results from their papers and adapt our TSPO to the same settings. As shown in Tab. 3, TSPO achieves enhanced performance despite using less training data.

Ablation of Training Data. Tab. 4 ablates our training data curation and reward mechanisms. First, directly using "Comprehensive Temporal" data improves performance by 3.9% on LongVideoBench and 1.1% on VideoMME. Next, the "Needle-in-a-Haystack" pipeline boosts LongVideoBench results yet degrades VideoMME performance, as LongVideoBench focuses on long-range temporal localization while VideoMME emphasizes general comprehension. Combining dual-style data achieves the best performance. For the ablation of rewards, using only the accuracy reward guides the temporal agent to select frames that yield correct answers, yet the supervision remains indirect. The temporal reward helps locate relevant clips with coarse labels, yet it may still include irrelevant frames. Combining both rewards enables the model to effectively locate the most relevant frames, leading to correct MLLM answers.

Exploring SFT\* for Keyframe. We investigate the possibility of end-to-end (E2E) optimization for the keyframe selector through SFT and compare it with TSPO. We utilize the Gumbel-Softmax technique (Wei et al. 2023), which enables the selected vision token to have gradients (Liang et al. 2024) and can be E2E trained. For SFT\* training, we randomly select 30K samples from LLaVA-Video-178K while keeping the MLLM parameters frozen. As shown in Tab. 5, our experimental results demonstrate that TSPO consistently outperforms SFT\*, which indicates TSPO's advantages, including its capacity to explore diverse sampling strategies

![](_page_6_Figure_7.jpeg)

Figure 5: Visualization comparisons of sampled frames and corresponding responses between ours and LLaVA-Video.

and utilize more direct reward signals.

Inference Efficiency. In this study, we fix the candidate frame number  $V_c$  to 128 (1FPS in our main setting), which avoids the effect of varying video lengths. As shown in Tab. 6, our efficiency can be demonstrated from the following aspects: 1) with the same 64 frames, ours achieves a consistent performance gain over the baseline; 2) with fewer yet informative frames, we maintain gains over baseline while requiring only half the number of tokens and reducing the LLM time to 50% of the original one; 3) for keyframe extraction time, our approach saves 90% of the time compared to CoS (Hu et al. 2025a), yet achieves performance gain. This demonstrates our efficiency in handling long videos.

Qualitative Results. Fig. 5 demonstrates the visualization results of keyframe selection and model responses. Our trained temporal agent is shown to achieve two capabilities: basic object recognition, *e.g.*, "short-haired woman" or "airplane", and temporal event relationship comprehension, *e.g.*, "entering the museum" or "appeared before". When the short-haired woman is localized, preceding contextual frames (the first scenes that the woman visits) are simultaneously captured, which confirms that our TSPO effectively guides the temporal agent to learn complex query-event correlation capacity. Furthermore, our precise keyframe selection enables the MLLM to generate accurate responses.

#### Conclusion

This paper proposes a Temporal Sampling Policy Optimization framework, which addresses the unsupervised and non-differentiable challenge of sparse frame sampling in Video-MLLMs. We propose an RL framework to optimize sparse frame sampling in an end-to-end manner, and propose a TSPO-targeted training data construction pipeline. Extensive comparison experiments and ablation studies validate the effectiveness and generalizability of our method.

