# A. Overview of Baselines

- FastV[\[8\]](#page-8-10) is a plug-and-play method that optimizes inference efficiency in LMMs by dynamically pruning visual tokens after the second layer, significantly reducing computational costs while maintaining performance. It identifies that image tokens receive drastically lower attention in LLM and strategically removes less impactful tokens.
- FitPrune [\[45\]](#page-10-5) is a training-free method for pruning visual tokens in multimodal LMMs, based on quickly estimating optimal pruning schemes through attention distribution fitting. It statistically determines which tokens can be discarded by minimizing divergence between attention distributions before and after pruning, using only a small batch of inference data. This approach rapidly produces a pruning recipe tailored to a given computation budget, significantly reducing computational complexity while preserving model performance.
- Pdrop [\[39\]](#page-9-7) accelerates large vision-language models by progressively removing redundant visual tokens in deeper layers based on token similarity. It partitions models into multiple stages, maintaining all tokens initially to preserve critical visual information, then gradually pruning tokens as layers deepen. This approach effectively reduces computational costs without compromising performance during both training and inference.
- Sparsevlm [\[55\]](#page-10-4) introduces a training-free, text-guided visual token sparsification method for LMMs, significantly reducing computational overhead by adaptively selecting important visual tokens based on relevant text prompts. It employs an adaptive pruning strategy at each layer and recycles pruned visual tokens into compact representations to minimize information loss.
- Visionzip [\[43\]](#page-10-2) is a simple yet effective method that reduces visual token redundancy in LMMs by selecting only the most informative tokens, significantly improving efficiency while maintaining performance. It employs a text-agnostic approach that merges and compresses redundant tokens, reducing computational costs and enhancing inference speed without requiring additional training.

## B. Overview of Benchmarks

• MME [\[11\]](#page-8-15) offers a robust benchmark for evaluating LVLMs across multimodal tasks. It assesses models on two major fronts: perception and cognition, using 14 well-structured subtasks that challenge their interpretive

- and analytical abilities.
- MMBench [\[28\]](#page-9-19) takes a two-pronged approach by introducing an extensive dataset that broadens the scope of evaluation questions and a novel CircularEval strategy that utilizes ChatGPT to convert free-form responses into structured answer choices.
- ScienceQA [\[29\]](#page-9-15) focuses on evaluating multi-hop reasoning and interpretability within scientific domains. It features a large dataset of approximately 21K multiplechoice questions across a variety of science topics, accompanied by detailed annotations and explanations.
- VizWiz [\[15\]](#page-8-14) stands out in the VQA field by using a dataset of over 31,000 visual questions that come from a real-world setting, featuring images taken by visually impaired individuals and their associated spoken queries, along with crowdsourced answers.
- GQA [\[1\]](#page-8-13) is built for complex visual reasoning tasks, containing 22 million questions generated from scene graphbased structures. It incorporates innovative evaluation metrics focused on consistency, grounding, and plausibility, pushing the boundaries of vision-language evaluation.
- POPE [\[21\]](#page-9-18) introduces a methodology to evaluate object hallucination in LVLMs, transforming the task into a binary classification problem. By using simple Yes-or-No prompts, POPE highlights model tendencies towards hallucination through various object sampling strategies.
- VQA [\[13\]](#page-8-18) collects complementary images such that every question in the balanced dataset is associated with a pair of similar images that result in two different answers to the question.
- ChartQA [\[30\]](#page-9-17) is a large-scale benchmark designed for question answering on charts, focusing on both visual and logical reasoning with 9.6K human-written and 23.1K automatically generated questions.
- DocVQA [\[31\]](#page-9-20) is a large-scale dataset designed for Visual Question Answering (VQA) on document images, containing 50,000 questions over 12,000+ real-world documents. Unlike previous datasets, it requires models to understand both textual content and visual layout, including tables, forms, and complex structures.
- MMstar [\[7\]](#page-8-7) is a new benchmark designed to address issues in evaluating Large Vision-Language Models (LVLMs), specifically unnecessary visual content and unintentional data leakage, which can mislead performance assessments. It includes 1,500 carefully selected vision-dependent samples, ensuring accurate evaluation of LVLMs' true multi-modal reasoning abilities. MMStar

<span id="page-12-0"></span>

| Methods                              | Token<br>Reduction | FLOPs ↓ (T)          | Δ                          | Latency ↓ (ms)          | Δ                          | KV Cache ↓<br>(MB)      | Δ                          | Performance ↑ | Δ                       |
|--------------------------------------|--------------------|----------------------|----------------------------|-------------------------|----------------------------|-------------------------|----------------------------|---------------|-------------------------|
| LLaVA-OneVision-7B                   | -                  | 71.4                 | -                          | 1040.1                  | -                          | 1786.4                  | -                          | 1581          | -                       |
| + VFlowOpt<br>+ FastV<br>+ VisionZip | 50%<br>50%<br>50%  | 37.2<br>38.1<br>37.7 | -48.0%<br>-46.6%<br>-47.2% | 584.2<br>615.1<br>580.7 | -43.8%<br>-41.9%<br>-44,2% | 902.8<br>902.8<br>902.8 | -49.5%<br>-49.5%<br>-49.5% | 1549          | +0.6%<br>-2.0%<br>+0.1% |

<span id="page-12-2"></span>Table 7. Efficiency analysis of LLaVA-OneVision-7B with VFlowOpt, FastV, and VisionZip. The detailed metric includes computation (FLOPs), latency, and KV-Cache memory. ( $\Delta$ ) denotes the reduction ratio.

|                       | MMStar | MME  | MMB  | SQA  | POPE | GQA  | DocVQA | VQA <sup>Text</sup> |
|-----------------------|--------|------|------|------|------|------|--------|---------------------|
| VisionZip             | 54.6   | 1562 | 78.9 | 90.4 | 88.8 | 61.0 | 79.6   | 70.0                |
| Ours (Random)         | 57.8   | 1570 | 79.9 | 92.3 | 89.1 | 61.2 | 82.3   | 72.5                |
| Ours (MathV360K-GEOS) | 57.8   | 1566 | 79.8 | 92.0 | 89.1 | 61.0 | 82.1   | 72.8                |

Table 8. Impact of optimization data selection

introduces new metrics—Multi-Modal Gain (MG) and Multi-Modal Leakage (ML)—to measure actual improvements from multi-modal training, with evaluations showing GPT-4V leading in both accuracy and multi-modal efficiency.

- SeedBench [18] is a large-scale benchmark designed to evaluate the generative comprehension capabilities of Multimodal Large Language Models (MLLMs), featuring 19K human-annotated multiple-choice questions across 12 evaluation dimensions for both images and videos.
- VideoMME [12] is the first comprehensive benchmark designed to evaluate Multi-Modal Large Language Models (MLLMs) in video analysis, covering 900 manually annotated videos across six diverse domains and 30 subcategories. It introduces a full-spectrum evaluation with multi-modal inputs, including subtitles and audio, and assesses models across various temporal contexts, from short clips to hour-long videos.

#### C. Efficiency Analysis about Baselines

We evaluate VFlowOpt, the well-performing baseline FastV, and VisionZip on efficiency metrics under the condition of retaining 50% of the tokens. With the same token retention rate, all methods showed identical KV-Cache memory usage, while FLOPs and latency exhibited slight differences, as shown in Tab. 7.

#### D. More ablation studies

## D.1. Choice of the optimization target

We are inspired by previous interpretability studies (Main Paper L273–L281) and consider the last token as the most representative one of such interactions. Results (shown in Tab. 9) show that optimizing for the last token yields the best performance. We will add this in the revised paper.

<span id="page-12-1"></span>

|              | MMStar | MME  | MMB  | SQA  | POPE | GQA  |
|--------------|--------|------|------|------|------|------|
| Last Token   | 57.8   | 1570 | 79.9 | 92.3 | 89.1 | 61.2 |
| Mean Pooling | 56.1   | 1549 | 77.5 | 92.1 | 88.5 | 60.6 |
| First Token  | 54.2   | 1530 | 77.7 | 89.5 | 85.4 | 60.4 |
| Top-3 Tokens | 56.8   | 1544 | 78.6 | 92.3 | 88.3 | 61.1 |

Table 9. Analysis of choice of the optimization target

<span id="page-12-3"></span>

|                            | DocVQA | VQA <sup>Text</sup> | POPE |
|----------------------------|--------|---------------------|------|
| VFlowOpt                   | 82.3   | 72.5                | 89.1 |
| w/o Importance Calibration | 80.3   | 71.4                | 88.6 |
| w/o Token Merging          | 82.0   | 72.4                | 86.8 |
| w/o Progressive Pruning    | 81.9   | 71.6                | 88.2 |

Table 10. Ablation studies on more benchmarks

#### D.2. Impact of optimization data selection

The result of our optimization is independent of data selection because the visual information flow being optimized is task-agnostic and model-specific. In our experiments, repeated random sampling yields nearly identical results. To further validate this, we optimize using 30 samples from the task-specific split (MathV360K-Geometry3K) of the LLaVA-OV training data. The model consistently achieves strong results across various tasks, regardless of data selection (shown in Tab. 8).

#### D.3. Ablation studies on more benchmarks

Additional results in Tab. 10 show that Token Merging is crucial for preserving coarse-grained semantics, while Importance Calibration and Progressive Pruning help maintain fine-grained visual perception.