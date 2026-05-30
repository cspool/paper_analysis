# <span id="page-28-1"></span>**B. Implementation Details**

### **B.1. Training Data**

**Data Composition.** We construct the training corpus from the difficulty-filtered VideoAuto-R1 [\(Liu et al.,](#page-20-3) [2026\)](#page-20-3) dataset, strictly retaining image and video samples while discarding pure-text examples. To guarantee robust coverage of visually demanding subdomains, we inject 16,500 high-complexity video instances from Video-R1 [\(Feng et al.,](#page-18-10) [2025\)](#page-18-10), prioritizing OCR, free-form QA, and regression-style tasks. The finalized pool comprises approximately 93.4K training samples. We rigorously purge all evaluation examples from this corpus to preclude data leakage.

### **B.2. Training Configuration**

We train the models for one epoch using AdamW with a global batch size of 128. We apply a learning rate of 2 × 10−<sup>5</sup> to the Allocator and 1 × 10−<sup>6</sup> to the backbone. We enforce a weight decay of 0.01 and cap gradient clipping at 1.0. We constrain the maximum video token budget to 8,192, sample *T*=128 frames during training, and bound the scale factors within [*s*min,*s*max] = [0.2, 1.8]. This range explicitly enables both aggressive downscaling and selective upscaling. CAPO samples *M*=16 allocation trajectories per prompt and executes *N*=1 rollout per trajectory. We orchestrate training across 32 H100 GPUs utilizing VeRL [\(Sheng](#page-21-13) [et al.,](#page-21-13) [2025\)](#page-21-13), DeepSpeed [\(Rasley et al.,](#page-20-12) [2020\)](#page-20-12), and vLLM [\(Kwon et al.,](#page-19-2) [2023\)](#page-19-2). We execute evaluations via lmms-eval [\(Zhang et al.,](#page-23-6) [2024a\)](#page-23-6). We truncate standard response lengths at 256 tokens and extend the limit to 4,096 tokens for reasoning models.

### <span id="page-28-0"></span>**B.3. Reward Design**

We detail the reward structures supplementing Sec. [3.3.](#page-5-3) The base scalar reward *R* task *m*,*n* isolates task-specific performance. Efficiency constraints manifest through CAPO advantage shaping rather than primitive additive reward terms.

**Base Task Reward** ( $R_{m,n}^{\text{task}}$ ). We define objective metrics for four task types:

- *Question Answering*. For math problems, we extract the numeric answer and tolerate a  $10^{-2}$  deviation from the ground truth. For multiple-choice questions, we parse the exact option letter. For standard QA, we execute exact string matching post-normalization (case-folding and whitespace stripping). This yields a binary reward  $R_{\text{QA}}(\hat{o}, o) \in \{0, 1\}$ .
- *Free-form Generation*. We quantify open-ended generation quality via the ROUGE-L score between the prediction  $\hat{o}$  and the reference o:  $R_{\text{Gen}}(\hat{o}, o) = \text{ROUGE-L}(\hat{o}, o) \in [0, 1]$ .
- Temporal Grounding. Let  $\mathcal{G} = \{[s_j, e_j]\}_j$  denote ground-truth segments and  $\widehat{\mathcal{G}} = \{[\hat{s}_k, \hat{e}_k]\}_k$  denote predictions. We isolate the highest temporal IoU across all pairs:  $R_{TG}(\widehat{\mathcal{G}}, \mathcal{G}) = \max_{[\hat{s}, \hat{e}] \in \widehat{\mathcal{G}}, [s, e] \in \mathcal{G}} \text{tIoU}([\hat{s}, \hat{e}], [s, e]) \in [0, 1]$ . Invalid parsings default to 0.
- *Grounding QA*. We parse both the textual response and temporal segments, summing their independent scores:  $R_{\text{GQA}}(\hat{o}, \widehat{\mathcal{G}}; o, \mathcal{G}) = R_{\text{QA}}(\hat{o}, o) + R_{\text{TG}}(\widehat{\mathcal{G}}, \mathcal{G}) \in [0, 2]$ .

These metrics establish the scalar base reward  $R_{m,n}^{task}$ . CAPO subsequently defines a binary success indicator  $u_{m,n} \in \{0,1\}$ . Exact-match QA utilizes the binary outcome directly. Continuous metrics (ROUGE-L, temporal IoU, Grounding QA) apply a strict 0.35 threshold. Format validation, when active, injects a weighted penalty prior to GRPO normalization, while  $u_{m,n}$  strictly isolates the task metric.

**Format Reward.** We enforce a binary format reward  $R_{\text{fint}}(\hat{o}) \in \{0,1\}$  via rigid regex validation. The generation must output exactly one <think>...</think> block and one <answer>...</answer> block. The final answer must reside within \\boxed{...} inside the <answer> tags. Malformed outputs trigger a penalty, integrating into the scalar reward with a 0.2 weight.

### **B.4. Prompt Template**

We deploy the standard GRPO training prompt (Table 6). The model must encapsulate its reasoning trace within <think> </think> tags. While optional for ResAdapt (as the MLLM  $\pi_{\phi}$  internalizes reasoning), we mandate this structure to ensure parity with reasoning-based baselines. The final answer must emerge encased in \\boxed{}.

### C. Extended Analysis

This section extends the analysis of Sec. 4.4. We first examine the learned allocation policy at per-benchmark, per-duration, and per-category granularity (Sec. C.1), then present extended ablations on the temporal regularizer and reward design (Sec. C.2), qualitative case studies linking allocation to reasoning outcomes (Sec. C.3), and a boundary-case transfer test beyond video (Sec. C.4). Unless otherwise noted, all plots use Qwen2.5-VL-7B processing 32 uniformly sampled frames.

### <span id="page-29-0"></span>C.1. Extended Scale Policy Analysis

**Benchmark-level budget allocation.** Figure 8 reveals a clear benchmark-level ordering, despite the policy operating without any knowledge of benchmark identity during training. Reasoning-intensive tasks consistently command higher mean scales than perception-oriented tasks (0.435 vs. 0.417). MMMU-Adaptation anchors the high-fidelity extreme, while VideoMME anchors the low-fidelity extreme. The policy does not

<span id="page-30-1"></span>**Table 6: Prompt template used for CAPO training.** The template presents video frames and the task question, requires intermediate reasoning inside <think> tags, and places the final answer in \boxed{} within <answer> tags. This structure enables automatic reward extraction from MLLM outputs.

