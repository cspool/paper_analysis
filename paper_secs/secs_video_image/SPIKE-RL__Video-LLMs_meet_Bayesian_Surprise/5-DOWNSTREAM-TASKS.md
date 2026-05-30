# 5 DOWNSTREAM TASKS

Having shown that SPIKE and SPIKE-RL can perform surprise localization, we now explore how identifying surprising segments of the video and allocating more frames to such regions can improve a Video-LLM's performance on downstream tasks as described in [§2.2.](#page-3-0)

#### 5.1 EXPERIMENTAL SETUP

Benchmarks. We evaluate our sampling method on a diverse selection of tasks, spanning surprise explanations, question answering, and temporal reasoning. The Reporter-MCQ portion of Black-SwanSuite [\(Chinchure et al., 2025\)](#page-9-5) tests models' ability to describe an unexpected event in a MCQ setup. FunQA's Task 2 [\(Xie et al., 2025\)](#page-12-1) and ExFunTube [\(Dayoon Ko, 2023\)](#page-9-6) ask models to generate an explanation of why events are surprising. Moving beyond surprising videos, we test our models on two MCQ tasks – VideoMME [\(Fu et al., 2024\)](#page-10-8), which probes general multimodal reasoning (we focus on short videos without subtitles), and NextQA [\(Xiao et al., 2021\)](#page-12-2), which targets temporal, commonsense, and causal reasoning.

Metrics. Following prior work [\(Majumdar et al., 2024;](#page-11-7) [Xie et al., 2025\)](#page-12-1), we evaluate the generative tasks using LLM-Match, prompting GPT-4o to rate the similarity between model-generated and ground-truth answers. Multiple-choice tasks are evaluated using accuracy.

Video-LLM Baselines. We consider widely adopted open-source Video-LLMs capable of video explanation and QA, including VideoChat2 [\(Li et al., 2024\)](#page-10-9), VideoLlama2 [\(Cheng et al., 2024\)](#page-9-3), and LLaVA-Video [\(Liu et al., 2023\)](#page-10-0). We also include FunMentor [\(Xie et al., 2025\)](#page-12-1), a model specifically designed for humor understanding. Our base model is Qwen2.5-VL (7B), which we use to evaluate

![](_page_7_Figure_0.jpeg)

<span id="page-7-1"></span>Figure 4: Qualitative Results. We compare Uniform, SPIKE and SPIKE-RL sampling methods. Errors in the explanation generated using uniform sampling reduce with SPIKE and are resolved with SPIKE-RL. We show belief hypotheses sets (Bt) at various timesteps, and observe how the divergence of Pprior and Ppost accurately captures the surprising moment in the video.

alternative sampling strategies under a fixed frame budget on BlackSwan and VideoMME-S. Finally, we test whether SPIKE-RL improves performance on a larger model, Qwen2.5-VL (32B).

Query-free Frame Sampling Baselines. To assess the effectiveness of our sampling, we benchmark against shot boundary detection methods on BlackSwan and Video-MME-S. Specifically, we tested RGB Histogram differences [\(V & Narayanan, 2015\)](#page-11-8), Edge Change Ratio (ECR; [Mann &](#page-11-9) [Kaur, 2015\)](#page-11-9), and motion-based detection [\(Wolf, 1996\)](#page-11-10), which capture changes in texture, structure, motion, and similarity. In all of these approaches, salient peaks are detected via smoothed scores and frames are distributed proportionally to peak strength, ensuring that the frame budget F is met. We also include Katna,[4](#page-7-0) a clustering-based method which applies K-means to frame histograms and selects the frame closest to each centroid. We use a maximum frame budget F of 64 frames for all our baselines, regardless of the sampling approach.

### 5.2 RESULTS

Table [2](#page-6-0) shows the performance of SPIKE and SPIKE-RL on downstream benchmarks. On tasks with surprising videos (BlackSwan, FunQA, ExFunTube), surprise-aware sampling provides substantial gains over uniform selection. Relative to uniform sampling, SPIKE improves accuracy by +1.6% on BlackSwan, +3.5% on FunQA, and +4.5% on ExFunTube. We observe that SPIKE-RL further extends performance on these tasks, with gains of +2.3% and +4.6% on BlackSwan and FunQA, and +7.0% on ExFunTube, marking our largest gains over uniform sampling. These results not only show the effectiveness of SPIKE in prioritizing surprising frames, but also credit the improved hypothesis quality in SPIKE-RL. On Qwen2.5-VL 32B, we see 2.3%, 3.1% and 3.9% gains respectively with SPIKE-RL, showing that our methods benefit larger models as well, extending their video understanding capability.

In general QA tasks (VideoMME-S, NextQA), we see moderate but consistent improvements over uniform sampling. SPIKE boosts scores by +1.0% on VideoMME-S and +1.2% on NextQA, while SPIKE-RL achieves +2.7% and +1.7% respectively on the 7B variant. The 32B variant with

<span id="page-7-0"></span><sup>4</sup><https://github.com/keplerlab/katna>

SPIKE-RL shows larger improvements of 3.6% and 1.8% on these tasks. These results show that surprise-aware sampling is broadly beneficial.

SBD strategies such as RGB Histogram, ECR, Katna, and Optical Flow consistently underperform uniform sampling. Their reliance on raw visual change makes them sensitive to camera motion and scene cuts, which rarely align with semantically important events. In contrast, our method offers principled guidance for identifying critical moments. Overall, we demonstrate that Bayesian Surprise provides a powerful inductive signal for adaptive frame selection: SPIKE delivers immediate gains by reallocating a fixed frame budget toward more informative segments, while SPIKE-RL further improves robustness through reinforcement-guided belief optimization.

#### 5.3 QUALITATIVE EXAMPLE

Figure [4](#page-7-1) illustrates the differences between uniform sampling, SPIKE, and SPIKE-RL. Under uniform sampling, the Video-LLM generates a caption that notes someone falling off a segway but misidentifies the person and the actions of the other riders (error highlighted in red). With the same frame budget, SPIKE and SPIKE-RL reallocate samples toward segments with high surprise scores, guided by observed belief shifts as demonsrated by the hypotheses. SPIKE correctly captures that the woman in the pink shirt and helmet loses balance and falls, though it still makes an error by stating that the other riders continue without stopping. SPIKE-RL improves on this. By more accurately localizing surprising segments – with one peak at the main fall and another smaller peak later – SPIKE-RL increases sampling density around both critical events. This leads to a more precise description of both the fall and the subsequent reactions of the other riders.

