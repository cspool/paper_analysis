# 4 Experiments

#### <span id="page-5-1"></span>4.1 Experimental Settings

**Benchmarks.** We evaluate our approach on four spatial reasoning benchmarks. **VSP** [Wu et al., 2024] measures spatial planning in a simulated maze-navigation environment. In addition to its main task, we adopt its spatial reasoning subtask, which asks the model to predict the outcome of a prescribed action sequence. We extend the original binary choice to a three-way classification. **BLINK-Jigsaw** [Fu et al., 2024] systematically evaluates the capacity of multimodal large language models to extrapolate global structural and semantic information from incomplete visual inputs, thereby assessing their proficiency in reasoning about spatial organization and maintaining perceptual coherence at a fine-grained level. **SAT** [Ray et al., 2024] evaluates both static and dynamic spatial relations. Additionally, we include the Mathematical Geometry subset of the recent **COMT** [Cheng et al., 2025b] to assess formal spatial reasoning in mathematical contexts. Full dataset details are provided in the supplementary material.

**Data Synthesis.** For each task, we sample 1k training instances for fine-tuning and 2k instances for reinforcement learning. COMT uniquely provides interleaved multimodal reasoning trajectories, which we directly use as both helper images and reasoning supervision. For the other benchmarks, we synthesize helper images and reasoning thoughts following the procedure outlined in Sec. 3.1. For VSP, the helper image is either the start map annotated with the red-arrow path (planning task) or the agent's current state snapshot (reasoning subtask). In Jigsaw, we concatenate one candidate patch beside the reference image. For SAT, we prompt a powerful video generation model CogVideoX-5B [Yang et al., 2024b] to render a scene that matches the textual description. With the generated helper image, we then employ Qwen2.5-VL 32B [Bai et al., 2025] as the external reasoning model  $M_r$  to generate textual thoughts. Specifically, three distinct reasoning trajectories are generated per helper image to encourage diversity in model outputs. Full synthesis details are provided in the supplementary material.

Table 1: Experimental Results on Visual-Spatial Planning (VSP) tasks.

<span id="page-6-2"></span><span id="page-6-1"></span>

| VSP            |         | Spati   | al Reason | ing     |      |         | Spat    | Spatial Planning |         |      |
|----------------|---------|---------|-----------|---------|------|---------|---------|------------------|---------|------|
|                | Level 3 | Level 4 | Level 5   | Level 6 | Avg. | Level 3 | Level 4 | Level 5          | Level 6 | Avg. |
| Zero-Shot      | 0.32    | 0.23    | 0.40      | 0.32    | 0.32 | 0.10    | 0.08    | 0.05             | 0.01    | 0.06 |
| Direct SFT     | 0.83    | 0.81    | 0.85      | 0.86    | 0.83 | 0.88    | 0.81    | 0.73             | 0.47    | 0.72 |
| CoT SFT        | 0.88    | 0.86    | 0.80      | 0.83    | 0.84 | 0.68    | 0.53    | 0.35             | 0.31    | 0.47 |
| GRPO           | 0.54    | 0.49    | 0.76      | 0.67    | 0.62 | 0.42    | 0.35    | 0.26             | 0.08    | 0.28 |
| CoT SFT + GRPO | 0.89    | 0.85    | 0.84      | 0.8     | 0.85 | 0.65    | 0.58    | 0.43             | 0.38    | 0.51 |
| Anole          | 0.46    | 0.51    | 0.49      | 0.63    | 0.52 | 0.02    | 0.01    | 0.00             | 0.00    | 0.01 |
| MVoT           | 0.53    | 0.64    | 0.67      | 0.59    | 0.61 | 0.21    | 0.11    | 0.08             | 0.03    | 0.11 |
| Ours (Direct)  | 0.86    | 0.84    | 0.88      | 0.87    | 0.86 | 0.93    | 0.83    | 0.76             | 0.51    | 0.76 |
| Ours (CoT)     | 0.87    | 0.92    | 0.86      | 0.84    | 0.87 | 0.75    | 0.63    | 0.53             | 0.39    | 0.58 |
| + w/ GRPO      | 0.92    | 0.90    | 0.86      | 0.88    | 0.89 | 0.78    | 0.65    | 0.52             | 0.43    | 0.60 |

**Baselines.** We compare our approach against both text-only baselines and recent unified multimodal models. First, we fine-tune the model directly using answer labels and also evaluate zero-shot reinforcement learning without any supervised warm-up. Next, using our synthetic data, we perform chain-of-thought supervised fine-tuning (CoT SFT) and then add reinforcement learning, giving a fair comparison. In addition, we benchmark against a unified model **Anole** [Chern et al., 2024], training with the same multimodal supervision, and **MVoT** [Li et al., 2025a], which generates action and state images but does not incorporate explicit reasoning thoughts during training.

Implementation Details. In this work, unless stated otherwise, all experiments use Qwen2.5-VL 7B as the base model. We perform supervised fine-tuning using a batch size of 8 and a cosine learning rate scheduler with an initial learning rate of 1e-5 for both stages. The random seed is fixed at 42 to ensure reproducibility. Reinforcement learning is implemented with the Verl framework. Unless stated otherwise, we use a latent token size of k=4 and a loss coefficient of  $\gamma=0.1$ .

#### <span id="page-6-0"></span>4.2 Experimental Results

We first evaluate the effectiveness of our method on the VSP benchmark. The results are shown in Tab. 1. We highlight the following findings. First, adding latent visual tokens to the reasoning process significantly improves the reasoning capability of VLMs compared to text-only baselines. Compared to directly fine-tuning the VLM with the synthesized data, our method achieves 3% higher accuracy on the spatial reasoning task and 11% on the spatial planning task. Also, with our two-stage training, Mirage improves the CoT SFT + GRPO, by 2% and 7%, respectively. This demonstrates the effectiveness of the proposed two-stage training method. Also, we test our method on COMT, Jigsaw, and SAT tasks and present the results in Tab. 2, where we observe the consistent performance gains on both tasks, underscoring that interleaving compact visual cues consistently strengthens spatial reasoning ability.

Additionally, we observe that unified model-based baselines such as MVoT and Anole, despite explicitly generating image tokens, perform poorly when faced with text and image interleave reasoning. After fine-tuning on the same data, they achieve only 61% accuracy on the spatial reasoning task and 11% on the spatial planning task. Notably, Anole struggles to even generate valid answers for the spatial planning task post fine-tuning. Following the setup in Li et al. [2025a], we construct interleaved reasoning trajectories by combining textual thoughts with simulated state images after each action step for the spatial reasoning task. While our reproduced results are lower than those reported in their paper, we attribute this discrepancy to the difference in training data. They use 6,846 samples, whereas we training with the same 1,000 samples to ensure a fair comparison. Even when compared to their reported results, our model still gains an additional 2% improvement. These findings further underscore the advantage of our latent design over current unified approaches.

We notice that on VSP spatial planning task, fine-tuning with synthesized reasoning thoughts performs significantly worse than training directly on answer labels, both with and without our latent design. Two factors likely contribute to this outcome. First, as noted in prior work [Li et al., 2025b], certain visual tasks that rely heavily on perception may not benefit from explicit reasoning during fine-tuning. Second, the synthesized thoughts are generated by Qwen2.5-VL-32B; although generally sound, they are not flawless, and any imperfections propagate into the base model. Likely, in SAT, the

Table 2: Experimental Results on COMT, Jigsaw, and SAT tasks.

<span id="page-7-3"></span><span id="page-7-0"></span>

| Method     | COMT | Jigsaw |         | SAT Synthetic |      | SAT Real |
|------------|------|--------|---------|---------------|------|----------|
|            |      |        | GoalAim | ObjM          | Avg. |          |
| Zero-Shot  | 0.63 | 0.58   | 0.50    | 0.63          | 0.57 | 0.49     |
| Direct SFT | 0.71 | 0.87   | 0.95    | 0.95          | 0.95 | 0.67     |
| CoT SFT    | 0.75 | 0.83   | 0.97    | 0.90          | 0.94 | 0.66     |
| GRPO       | -    | 0.85   | 0.85    | 0.80          | 0.83 | 0.71     |
| SFT + GRPO | -    | 0.86   | 0.93    | 0.85          | 0.89 | 0.65     |
| Ours       | 0.77 | 0.88   | 0.98    | 0.98          | 0.98 | 0.72     |

<span id="page-7-2"></span>Table 3: Experimental Results with Qwen2.5-VL 3B on COMT, Jigsaw, and SAT tasks.

| Method     | COMT | Jigsaw | SAT Synthetic |      |      | SAT Real |
|------------|------|--------|---------------|------|------|----------|
|            |      |        | GoalAim       | ObjM | Avg. |          |
| Zero-Shot  | 0.40 | 0.45   | 0.50          | 0.38 | 0.44 | 0.51     |
| Direct SFT | 0.67 | 0.80   | 0.82          | 0.83 | 0.83 | 0.55     |
| CoT SFT    | 0.65 | 0.59   | 0.73          | 0.88 | 0.71 | 0.54     |
| GRPO       | -    | 0.54   | 0.78          | 0.80 | 0.79 | 0.54     |
| SFT + GRPO | -    | 0.72   | 0.82          | 0.85 | 0.84 | 0.52     |
| Ours       | 0.68 | 0.85   | 0.85          | 0.93 | 0.89 | 0.64     |

helper images are produced by a video generation model without ground-truth annotations, which can introduce further noise to the latent prior. Despite these challenges, our latent reasoning pipeline still closes much of the performance gap, highlighting its practical robustness.

Moreover, reinforcement learning can further improve the performance of our method. As shown in Tab. [1,](#page-6-1) by weaving latent visual tokens within the text trajectories, instead of placing them at the start, our model can naturally explore diverse sequences. After optimizing with GRPO, Mirage achieves extra gains (+2% accuracy) on VSP tasks. These results further confirm that interleaved latent cues provide informative guidance with flexible reasoning, highlighting the potential of our latent design.

