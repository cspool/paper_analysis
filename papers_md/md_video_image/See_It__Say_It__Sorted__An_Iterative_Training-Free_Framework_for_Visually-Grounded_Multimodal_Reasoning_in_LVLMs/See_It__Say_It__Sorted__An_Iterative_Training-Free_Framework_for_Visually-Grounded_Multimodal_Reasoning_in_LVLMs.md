# <span id="page-0-1"></span>See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

Yongchang Zhang<sup>1,3\*</sup>, Oliver Ma<sup>2\*</sup>, Tianyi Liu<sup>1†</sup>, Guangquan Zhou<sup>1†</sup>, Yang Chen<sup>1†</sup> Southeast University <sup>2</sup>University of Oxford <sup>3</sup>AIIA, Ministry of Education, China

yongchangzhang2005@gmail.com chenyang.list@seu.edu.cn

https://github.com/uuuuZYC/See-It-Say-It-Sorted

## **Abstract**

Recent large vision-language models (LVLMs) have demonstrated impressive reasoning ability by generating long chain-of-thought (CoT) responses. However, CoT reasoning in multimodal contexts is highly vulnerable to visual hallucination propagation: once an intermediate reasoning step becomes inconsistent with the visual evidence, subsequent steps—even if logically valid—can still lead to incorrect final answers. Existing solutions attempt to mitigate this issue by training models to "think with images" via reinforcement learning (RL). While effective, these methods are costly, model-specific, and difficult to generalize across architectures. Differently, we present a lightweight method that bypasses RL training and provides an iterative, training-free, plug-and-play framework for visuallygrounded multimodal reasoning. Our key idea is to supervise each reasoning step at test time with visual evidence, ensuring that every decoded token is justified by corresponding visual cues. Concretely, we construct a textual visual-evidence pool that guides the model's reasoning generation. When existing evidence is insufficient, a visual decider module dynamically extracts additional relevant evidence from the image based on the ongoing reasoning context, expanding the pool until the model achieves sufficient visual certainty to terminate reasoning and produce the final answer. Extensive experiments on multiple LVLM backbones and benchmarks demonstrate the effectiveness of our approach. Our method achieves 16.5%-29.5% improvements on TreeBench and 13.7% RH-AUC gains on RH-Bench, substantially reducing hallucination rates while improving reasoning accuracy without additional training.

## 1. Introduction

Large vision—language models (LVLMs) now generate long chain-of-thought (CoT) explanations and solve diverse mul-

<span id="page-0-0"></span>![](_page_0_Figure_10.jpeg)

Figure 1. **Reasoning pattern comparison.** (a) Greedy decoding: the base VLM selects the top-1 token at each step; any hallucination in an intermediate step propagates to an incorrect final answer. (b) RLHF-based "think-with-images": the model learns when to call tools to zoom or crop the image and re-inject cropped regions into the reasoning context—effective but costly and model-specific. (c) Ours: a lightweight, training-free, model-agnostic framework. A supervisor maintains a dynamic visual-evidence pool to detect and correct hallucination steps. When uncertainty arises, it invokes a visual decider to extract new evidence, enabling visually grounded reasoning throughout the chain.

timodal reasoning tasks [7, 18, 20, 22, 23]. Yet, the very ability to "think more" often coincides with "seeing less". [10] During inference-time decoding, a model must balance three competing contexts: the image, a growing textual context, and instruction tokens. As the context lengthens, subtle but decisive visual cues are easily dominated by language priors. Even a single token that departs from the visual evidence can steer the remaining chain of thought toward a fluent but visually inconsistent trajectory (Fig. 1 (a)). This reasoning—perception drift originates at decoding time—not because the model lacks visual understanding, but because long-horizon token generation gradually ampli-

<sup>\*</sup>Equal contribution †Corresponding author

<span id="page-1-0"></span>fies language priors over visual grounding [\[9,](#page-8-6) [11,](#page-8-7) [17,](#page-8-8) [25\]](#page-8-9).

A prevailing solution is to explicitly train models to "think with images" by learning when and where to zoom or crop during CoT generation (Fig. [1\(](#page-0-0)b)). These RL- or preference-optimization pipelines train a policy to call visual tools, inspect regions, and re-inject pixels into the reasoning context. While effective, they rely on curated data, reward design, heavy computation, and tight coupling to specific backbones. The learned policy intertwines spatial exploration with the CoT, repeatedly encoding cropped views and incurring latency. As model scale increases, such visually grounded training becomes prohibitively expensive, limiting accessibility for broader use.

To address these limitations, we pursue a different principle: rather than learning when to look at training time, we supervise each reasoning step with visual evidence at test time (Fig. [1](#page-0-0) (c)). We introduce an iterative, training-free, plug-and-play framework that treats decoding as a sequence of evidence-justified token selections. The system maintains a textual evidence pool that operates alongside the base LVLM. At each step, the LVLM proposes a compact top-k set of candidate tokens from its local probability distribution. A lightweight supervisor computes an evidenceinduced preference over these candidates and negotiates a reweighted distribution with the base probabilities, preserving confident behavior while reallocating probability mass toward tokens consistent with accumulated evidence. When residual uncertainty remains, a visual decider inspects the image under the current reasoning context and generates a concise micro-observation in natural language. This observation is appended to the evidence pool and reused in all subsequent reasoning steps.

This design exhibits three key properties that directly mitigate the limitations of RL-based pipelines such as PixelReasoner [\[19\]](#page-8-10) and DeepEyes [\[29\]](#page-9-0). First, it is training-free and inherently transferable: the framework wraps around a frozen LVLM with a lightweight decider, requiring no task-specific finetuning or policy optimization. Second, it is cost-aware by construction. A simple uncertainty test on the negotiated distribution determines whether to invoke the decider, ensuring that additional visual computation occurs only when most likely to prevent a hallucination. Third, it represents evidence in text rather than pixels, enabling subsequent tokens to directly reference prior micro-observations without repeatedly re-encoding image crops. This textual form makes the framework easier to deploy at inference time and substantially reduces computational overhead compared with pixel-level reasoning.

Empirically, the framework improves both grounding and end-task accuracy across backbones and benchmarks while keeping overhead modest. Because evidence is accumulated on demand, fine-grained cues can be reused downstream to stabilize the remainder of the chain. Qualitative analyses show that many previously cascading failures reduce to one or two decisive micro-observations that the decider contributes precisely at uncertain steps; Quantitatively, we observe a clear accuracy–latency trade-off as the uncertainty threshold varies, allowing flexible adaptation to different deployment budgets.

The main contributions are summarized as follows:

- We present a training-free, plug-and-play decoding framework that supervises token selection with a growing textual evidence pool and negotiates next-token probabilities with the base model rather than relying on learned tool-calling policies.
- An uncertainty-triggered visual decider emits concise, reusable micro-evidence only when necessary, yielding strong cost–accuracy trade-offs.
- The approach transfers across LVLM backbones and consistently reduces hallucination while obviously improving task accuracy on a wide range of benchmarks.

# 2. Methodology

## 2.1. Visual Description Grounded Decoding

Let a VLM define, at decoding step i, a next-token distribution:

$$p_i(w) = p_{\text{VLM}}(x_i = w \mid x_{< i}), \qquad w \in \mathcal{V}.$$
 (1)

Visual description grounded decoding (VDGD) first asks the model to produce one global textual description d = (d1, . . . , dL) of image and then constrains decoding by knee truncation to select top-k plausible tokens and a descriptionbased KL preference. If p(1) ≥ p(2) ≥ · · · are sorted probabilities of p<sup>i</sup> , the knee index can be defined as follows:

$$k^* = \arg\max_{k} (p_{(k)} - p_{(k+1)}).$$
 (2)

And then we can define the candidate set:

$$C_i = \text{Top-}k^*(p_i). \tag{3}$$

For a candidate token w ∈ C<sup>i</sup> and a prefix of description length j (1 ≤ j ≤ L), VDGD measures the deviation:

$$\mathrm{KL}\big(\mathrm{onehot}(w) \mid\mid p_{\mathrm{VLM}}(\cdot \mid d_{< j})\big) = -\log p_{\mathrm{VLM}}(w \mid d_{< j}),\tag{4}$$

then takes a minimum value over j (the best prefix) and replaces the base logits with these values before softmax. While effective in its one-shot setting, fixed logit replacement tied to a single static caption is less self-adaptive, and it does not preserve the base model's calibrated confidence on already-confident steps.

## 2.2. Distribution Supervisor

As shown in Fig. [2,](#page-2-0) The distribution supervisor keeps the spirit of description-guided preference yet makes two principled changes: (a) we convert VDGD's min-over-prefix KL into a mean-over-prefix probability that is further averaged across multiple evidences, and (b) we mix the

<span id="page-2-5"></span><span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 2. Overview of evidence-constrained reweighting decoding (ECRD) at decoding step i. The base VLM emits a top-k candidate set; the supervisor builds an evidence-induced distribution from the current evidence pool and negotiates with the base probabilities to reweight candidates. If confidence remains low, the visual decider reads the image with the current prefix, commits a token, and adds a short textual evidence for later steps.

evidence-induced distribution with the base model instead of overwriting logits.

From VDGD's KL to ECRD's evidence score. Let  $\mathcal{E}_i = (e_1, \dots, e_L)$  be a piece of evidence sentence. VDGD's min-over-prefix KL is well suited to the static captioning regime where the most supportive partial description is selected. In our dynamic regime, where evidence accrues incrementally over steps and will be mixed with the base distribution, thus, we need an aggregator that rewards sustained support across the sentence rather than one sharp peak without robustness, and compiles smoothly across multiple evidence. We therefore replace the min with a mean-over-prefix probability:

<span id="page-2-2"></span>
$$q_{\mathcal{E}}(w) = \frac{1}{L} \sum_{j=1}^{L} p_{\text{VLM}}(w \mid e_{< j}).$$
 (5)

Given a pool  $E_i = \{\mathcal{E}_1, \dots, \mathcal{E}_N\}$  of evidence available at step i, we average their supports:

<span id="page-2-4"></span>
$$S_i(w) = -\log\left(\frac{1}{N}\sum_{\mathcal{E}}q_{\mathcal{E}}(w)\right),$$
 (6)

and restrict to  $C_i$  to obtain the evidence-induced distribution:

<span id="page-2-3"></span>
$$r_i(w) = \begin{cases} \frac{\exp\{-S_i(w)\}}{\sum_{u \in \mathcal{C}_i} \exp\{-S_i(u)\}}, & w \in \mathcal{C}_i, \\ 0, & \text{otherwise.} \end{cases}$$
(7)

**Negotiated reweighting.** Let  $p_i$  be the base distribution. We form a mass-matched version  $\tilde{r}_i$ , the rationale is as follows:  $p_i$  and the evidence-induced distribution  $r_i$  are not on the same scale.  $r_i$  is normalized only over the candidate set  $\mathcal{C}_i$  (and zero elsewhere), whereas  $p_i$  spreads probability over the full vocabulary. We therefore perform a proportional

rescaling of  $r_i$  within  $C_i$  so that the total mass assigned to  $C_i$  matches that of  $p_i$ . The resulting scaled distribution is denoted  $\tilde{r}_i$ . Concretely,

$$\tilde{r}_{i}(w) = \begin{cases} r_{i}(w) \cdot \frac{\sum_{u \in \mathcal{C}_{i}} p_{i}(u)}{\sum_{u \in \mathcal{C}_{i}} r_{i}(u)}, & w \in \mathcal{C}_{i}, \\ 0, & \text{otherwise,} \end{cases}$$
(8)

which ensures that we only reallocate the probability mass within  $C_i$  without varying its total amount:

$$\sum_{w \in \mathcal{C}_i} \tilde{r}_i(w) = \sum_{w \in \mathcal{C}_i} p_i(w). \tag{9}$$

and then let the base distribution and the evidenceinduced distribution negotiate a mixture:

<span id="page-2-1"></span>
$$p_i^{\text{mix}}(w) = \begin{cases} \alpha_i \, p_i(w) + (1 - \alpha_i) \, \tilde{r}_i(w), & w \in \mathcal{C}_i, \\ \alpha_i \, p_i(w), & w \notin \mathcal{C}_i. \end{cases}$$
(10)

The adaptive weight is chosen without hyper-parameters as:

$$\alpha_i = p_{(1)}, \tag{11}$$

the top probability of the base model before mixing. This choice aligns with empirical statistics reported by [5]: hallucination steps tend to have larger knee-selected  $k^\star$  and diffuse local distributions with small variance. However, non-hallucination steps are sharply peaked with average  $k^\star\approx 1$ . Consequently, when  $p_{(1)}$  is large (confident step), we keep the base distribution predominates; when  $p_{(1)}$  is small (hallucination-critical moments), evidence receives more weight. It reflects a general intervention principle: when the base distribution is sharp, evidence should act as a light prior; when it is diffuse, evidence should carry more weight. This makes mixture responsive to local uncertainty while preserving the behavior of base model in easy steps.

<span id="page-3-1"></span>When to acquire new evidence. After Eq. [\(10\)](#page-2-1), we inspect the negotiated margin:

<span id="page-3-0"></span>
$$\Delta_i = p_{(1)}^{\text{mix}} - p_{(2)}^{\text{mix}}. \tag{12}$$

If k <sup>⋆</sup> > 1 and ∆<sup>i</sup> ≤ δ (δ is a hyperparameter), we consider step i a probable trigger for hallucination and acquire new evidence. Otherwise, we directly select p mix (1) .

## 2.3. Dynamic evidence pool and the visual decider

Invocation and interface. When the trigger in Eq. [\(12\)](#page-3-0) fires, we call a lightweight visual decider instantiated by GRIT [\[4\]](#page-8-12), which built on Qwen2.5-VL-3B [\[1\]](#page-8-13), except for the basic answer, it can optionally output the coordinates of one or more image regions that the model needs to refer to during its reasoning process when generating an answer. GRIT receives the image, the tail of the textual prefix, and the candidates C<sup>i</sup> . It should be noted that GRIT does not receive the original question at this stage, as we only aim to mitigate possible hallucinations occurring in the current step. The model is required to resolve the content of this step alone, rather than the complete question. We obtain (i) a choice w <sup>⋆</sup> ∈ C<sup>i</sup> , (ii) a single human-readable evidence sentence E<sup>i</sup> from its answer. We then force w ⋆ at step i and append the sentence to the evidence pool:

$$E_{i+1} \leftarrow E_i \cup \{\mathcal{E}_i\}. \tag{13}$$

Representation. Crucially, only the text participates in scoring via Eqs. [\(5\)](#page-2-2) to [\(7\)](#page-2-3), and coordinates are stored as annotations for interpretability and for binding the sentence to concrete regions but never participate in scoring steps. This design is deliberate. Injecting raw crops back into the context, such as in zoom-in pipelines, repeatedly couples pixel processing with the entire reasoning chain and typically requires additional supervision or preference optimization for the model to learn when and where to zoom. In contrast, textual evidence is compact, semantic, and model-native: it lives in the same token space as the decoder, can be scored without revisiting pixels, and naturally composes with prior sentences. This keeps the intervention lightweight while saving a verifiable trail.

Initialization and growth. We initialize E<sup>0</sup> with a global description dglobal to provide broad coverage but not to act as the sole evidence source. Thereafter the evidence pool grows only on demand. This maintains modest computation, preserves determinism in non-ambiguous regions, and lets evidence accumulate precisely where it is most useful for subsequent decisions.

Semantics and micro-views. Although textual, each sentence implicitly refers to one or multiple related subviews of the image, especially when GRIT generates the answer with coordinates. Because every sentence is generated to resolve a specific local choice, the evidence pool accumulates a set of semantically linked micro-observations that remain logically connected to the evolving chain of thought. Eqs. [\(6\)](#page-2-4) and [\(7\)](#page-2-3) convert these micro-observations into reusable probability mass, allowing later steps to benefit from earlier visual disambiguation without re-encoding crops. Therefore ECRD is a bridge that allows base model to interact with a collection of subviews organized by reasoning needs, rather than with isolated cropped regions, thereby enabling reuse across the whole sequence.

# 3. Experiment

# 3.1. Setups

We evaluate ECRD across benchmarks grouped by the capabilities they probe: (i) visual grounded reasoning under long chains, (ii) reasoning–perception balance and hallucination control, and (iii) broad multimodal competence. In the experiments reported in Tabs. [1](#page-4-0) to [5,](#page-6-0) we set the uncertainty threshold δ = 0.08. No additional training are used.

# 3.2. Benchmarks and base models

Benchmarks. We evaluate on three groups of benchmarks. TreeBench [\[24\]](#page-8-14) probes "thinking with images" by separating Perception (identify, localize, read, attribute) from Reasoning (operate over aggregated visual evidence such as occlusion, containment, ordering, perspective); the metric is answer accuracy. RH-Bench [\[10\]](#page-8-5) reports Reason and Perception scores and RH-AUC [\[10\]](#page-8-5), which summarizes the trade-off between reasoning length and hallucination (higher is better). For general multimodal competence, we use V\*Bench [\[27\]](#page-9-1), MathVista [\[13\]](#page-8-15), ChartQA [\[14\]](#page-8-16), OCR-Bench [\[12\]](#page-8-17), and HallusionBench [\[6\]](#page-8-18), evaluated by accuracy.

Base models. Unless otherwise specified, the visual decider is GRIT-3B [\[4\]](#page-8-12), which is built upon Qwen2.5-VL-3B [\[1\]](#page-8-13) and optimized for visual grounding. To assess plugand-play generality and scale robustness, we also attach ECRD without any finetuning to three open-source general model families: LLaVA-OneVision [\[8\]](#page-8-19) (7B and 72B), Qwen2.5-VL [\[1\]](#page-8-13) (7B, 32B and 72B), and InternVL3 [\[30\]](#page-9-2) (8B, 38B and 78B). All checkpoints and inference recipes follow the authors' official releases. ECRD modifies only the decoding procedure at test time and leaves the underlying encoders or decoders frozen. For reference lines on TreeBench, we also report private models GPT-4o [\[15\]](#page-8-20) and o3 [\[16\]](#page-8-21) as well as Gemini-2.5-Flash [\[2\]](#page-8-22) and Gemini-2.5- Pro [\[3\]](#page-8-23), and include recent RL-based visually-grounded reasoning systems DeepEyes [\[29\]](#page-9-0), Pixel-Reasoner [\[19\]](#page-8-10), and TreeVGR-7B [\[24\]](#page-8-14) for comparison. The paper that presents the TreeVGR-7B [\[24\]](#page-8-14) method also introduces TreeBench.

## 3.3. Main results and analysis

As demonstrated in Tab. [1,](#page-4-0) ECRD delivers consistent gains across all open-source general backbones and scales on TreeBench, showing that the method is truly plug-and-play

<span id="page-4-2"></span><span id="page-4-0"></span>

|                                              |                | Perception         |              |                            |                   |               | Reasoning  |                  |               |               |               |
|----------------------------------------------|----------------|--------------------|--------------|----------------------------|-------------------|---------------|------------|------------------|---------------|---------------|---------------|
|                                              | Overall        | Attr.              | Mater.       | Phys.                      | ObjRet.           | OCR           | Persp.     | Order.           | Cont.&Oc.     | Contain.      | Compar.       |
|                                              | Private Models |                    |              |                            |                   |               |            |                  |               |               |               |
| Gemini-2.5-Flash-0520 [2]                    | 45.9           | 48.3               | 53.9         | 69.6                       | 68.8              | 75.0          | 15.3       | 19.3             | 56.1          | 72.4          | 43.2          |
| GPT-4o-1120 [15]                             | 46.9           | 51.7               | 61.5         | 65.2                       | 43.8              | 69.1          | 18.8       | 38.6             | 48.8          | 72.4          | 43.2          |
| Gemini-2.5-Pro-0605 [3]                      | 54.1           | 51.7               | 61.5         | 56.5                       | 75.0              | 83.8          | 20.0       | 36.8             | 65.9          | 86.2          | 54.6          |
| o3-0416 [16]                                 | 54.8           | 69.0               | 69.2         | 65.2                       | 68.8              | 79.4          | 22.4       | 38.6             | 61.0          | 86.2          | 50.0          |
|                                              |                |                    |              | Open-source General Models |                   |               |            |                  |               |               |               |
| Qwen2.5-VL-7B [1]                            | 37.0           | 55.2               | 53.8         | 56.5                       | 62.5              | 27.9          | 20.0       | 35.1             | 39.0          | 44.8          | 43.2          |
| + ECRD                                       | 47.9↑          | 10.9 72.4↑<br>17.2 | 53.80.0      | 73.9↑<br>17.4              | 62.50.0           | 54.4↑<br>26.5 | 20.00.0    | 35.10.0          | 56.1↑<br>17.1 | 75.9↑<br>31.1 | 45.5↑<br>2.3  |
| Qwen2.5-VL-32B [1]                           | 42.5           | 51.7               | 53.8         | 69.6                       | 62.5              | 54.4          | 16.5       | 33.3             | 46.3          | 62.1          | 38.6          |
| + ECRD                                       | 48.6↑<br>6.1   | 62.1↑<br>10.4      | 53.80.0      | 73.9↑<br>4.3               | 62.50.0           | 60.3↑<br>5.9  | 23.5↑      | 7.0 35.1↑<br>1.8 | 61.0↑<br>14.7 | 65.5↑<br>3.4  | 45.5↑<br>6.9  |
| Qwen2.5-VL-72B [1]                           | 42.2           | 65.5               | 69.2         | 56.5                       | 56.3              | 48.5          | 11.8       | 33.3             | 51.2          | 72.4          | 38.6          |
| + ECRD                                       | 49.9↑<br>7.7   | 65.50.0            | 69.20.0      | 65.2↑<br>8.7               | 62.5↑             | 6.2 69.1↑     | 20.6 20.0↑ | 8.2 36.8↑<br>3.5 | 58.5↑<br>7.3  | 75.9↑<br>3.5  | 40.9↑<br>2.3  |
| LLaVA-OneVision-7B [8]                       | 37.3           | 55.2               | 53.8         | 56.5                       | 50.0              | 32.4          | 21.2       | 22.8             | 41.5          | 72.4          | 36.4          |
| + ECRD                                       | 43.5↑<br>6.2   | 55.20.0            | 61.5↑<br>7.7 | 60.9↑<br>4.4               | 56.3↑             | 6.3 47.1↑     | 14.7 23.5↑ | 2.3 28.0↑<br>5.2 | 53.7↑<br>12.2 | 72.40.0       | 40.9↑<br>4.5  |
| LLaVA-OneVision-72B [8]                      | 40.5           | 62.1               | 53.8         | 65.2                       | 62.5              | 36.8          | 12.9       | 28.1             | 53.7          | 65.5          | 47.7          |
| + ECRD                                       | 46.9↑<br>6.4   | 62.10.0            | 61.5↑<br>7.7 | 69.6↑<br>4.4               | 68.8↑             | 6.3 51.5↑     | 14.7 18.8↑ | 5.9 29.8↑<br>1.7 | 61.0↑<br>7.3  | 72.4↑<br>6.9  | 52.3↑<br>4.6  |
| InternVL3-8B [30]                            | 38.8           | 51.7               | 69.2         | 56.5                       | 56.3              | 33.7          | 21.2       | 24.6             | 39.0          | 72.4          | 43.2          |
| + ECRD                                       | 45.2↑<br>6.4   | 51.70.0            | 69.20.0      | 69.6↑                      | 13.1 62.5↑        | 6.2 50.0↑     | 16.3 24.7↑ | 3.5 29.8↑<br>5.2 | 43.9↑<br>4.9  | 72.40.0       | 50.0↑<br>6.8  |
| InternVL3-38B [30]                           | 42.0           | 51.7               | 61.5         | 52.2                       | 68.8              | 51.5          | 12.9       | 33.3             | 56.1          | 65.5          | 38.6          |
| + ECRD                                       | 48.4↑<br>6.4   | 51.70.0            | 69.2↑        | 7.7 65.2↑                  | 13.0 75.0↑<br>6.2 | 58.8↑<br>7.3  | 22.4↑      | 9.5 36.8↑<br>3.5 | 56.10.0       | 69.0↑<br>3.5  | 50.0↑<br>11.4 |
| InternVL3-78B [30]                           | 46.4           | 62.1               | 61.5         | 52.2                       | 68.8              | 52.9          | 16.5       | 33.3             | 61.0          | 86.2          | 45.5          |
| + ECRD                                       | 50.9↑<br>4.5   | 65.5↑<br>3.4       | 69.2↑        | 7.7 65.2↑                  | 13.0 75.0↑<br>6.2 | 60.3↑<br>7.4  | 21.2↑      | 4.7 35.1↑<br>1.8 | 63.4↑<br>2.4  | 86.20.0       | 47.7↑<br>2.2  |
| Open-source Visual Grounded Reasoning Models |                |                    |              |                            |                   |               |            |                  |               |               |               |
| DeepEyes-7B [29]                             | 37.5           | 62.1               | 53.8         | 65.2                       | 68.8              | 51.5          | 11.8       | 24.6             | 36.6          | 51.7          | 47.7          |
| Pixel-Reasoner-7B [19]                       | 39.0           | 58.6               | 61.5         | 65.2                       | 50.0              | 48.5          | 14.1       | 31.6             | 39.0          | 44.8          | 40.9          |
| TreeVGR-7B [24]                              | 50.4           | 65.5               | 53.8         | 82.6                       | 68.8              | 63.3          | 22.4       | 36.8             | 61.0          | 69.0          | 45.5          |

Table 1. Results of different models on TreeBench. Best performances for open-source models are highlighted in bold. ECRD consistently improves open-source general models across architectures and scales, confirming training-free plug-and-play applicability.

rather than model-specific. On Qwen2.5-VL-7B the overall accuracy rises from 37.0% to 47.9%, and similar improvements appear on LLaVA-OneVision-7B and InternVL3-8B, as well as on larger models such as Qwen2.5-VL-32B/72B, LLaVA-OneVision-72B, and InternVL3-38B/78B, where the absolute lifts are typically in the +4-8 point band. The sub-category pattern matches our design: competencies that rely on checkable visual facts such as OCR, Physical State and Comparison benefit the most. Compared to RLHF-based visual-grounded reasoning models, ECRD on Qwen2.5-VL-7B surpasses DeepEyes-7B and Pixel-Reasoner-7B and approaches TreeVGR-7B, yet it requires no additional training, curated traces, or reinforcement optimization. It also exceeds several strong closed models such as Gemini-2.5-Flash and GPT-4o while still trailing Gemini-2.5-Pro and o3.

To further position ECRD against established trainingfree baselines beyond RL-based systems, we compare it on TreeBench using the same base model (Qwen2.5-VL-7B) with: inference-time correction (Woodpecker [\[28\]](#page-9-3)), programmatic reasoning (ViperGPT [\[21\]](#page-8-24)), and visual prompting (ControlMLLM [\[26\]](#page-8-25)). We also evaluate standard decoding alternatives, including beam search, self-consistency,

<span id="page-4-1"></span>

|              |      |      | Method Base Woodpecker ViperGPT ControlMLLM Beam Self-cons. Diverse ECRD |      |      |      |      |
|--------------|------|------|--------------------------------------------------------------------------|------|------|------|------|
| Overall 37.0 | 36.3 | 38.3 | 40.5                                                                     | 38.8 | 39.0 | 38.3 | 47.9 |

Table 2. Comparison with training-free baselines and decoding alternatives on TreeBench. Best result are highlighted in bold. ECRD performs markedly better among all plug-and-play settings.

and diverse sampling. As shown in Tab. [2,](#page-4-1) all these baselines underperform ECRD, indicating a significantly better performance of ECRD among plug-and-play approaches.

Tab. [4](#page-5-0) shows that ECRD improves Reasoning from 39.6% to 46.4% and Perception from 50.2% to 57.1%, lifting RH-AUC from 0.51 to 0.58 on RH-Bench . A higher RH-AUC value indicates that as the chain grows longer, accuracy remains at a higher level and the balance between reasoning and hallucination is better maintained.

Finally, across five general multimodal benchmarks in Tab. [5,](#page-6-0) ECRD brings broad, task-general gains on both Qwen2.5-VL-7B and LLaVA-OneVision-7B: V\*Bench overall rises by several points, MathVista and ChartQA see steady improvements, OCRBench jumps by around +8–12 points, and HallusionBench improves by roughly +8–11 points, reflecting fewer visually induced slips that would otherwise propagate through the chain.

<span id="page-5-2"></span><span id="page-5-1"></span>

|                                                           |      | Perception |      |      |      |      | Reasoning |      |                                                                                 |      |      |
|-----------------------------------------------------------|------|------------|------|------|------|------|-----------|------|---------------------------------------------------------------------------------|------|------|
|                                                           |      |            |      |      |      |      |           |      | Overall Attr. Mater. Phys. ObjRet. OCR Persp. Order. Cont.&Oc. Contain. Compar. |      |      |
| GRIT-3B [4]                                               | 30.1 | 31.0       | 23.1 | 56.5 | 25.0 | 39.7 | 21.2      | 17.5 | 36.6                                                                            | 48.3 | 20.5 |
| Qwen2.5-VL-7B [1]                                         | 37.0 | 55.2       | 53.8 | 56.5 | 62.5 | 27.9 | 20.0      | 35.1 | 39.0                                                                            | 44.8 | 43.2 |
| Qwen2.5-VL-7B + VDGD                                      | 39.5 | 58.6       | 53.8 | 52.2 | 62.5 | 50.0 | 17.6      | 29.8 | 39.0                                                                            | 48.3 | 40.9 |
| Qwen2.5-VL-7B + supervisor                                | 40.7 | 58.6       | 53.8 | 56.5 | 62.5 | 51.5 | 18.8      | 31.6 | 41.5                                                                            | 44.8 | 43.2 |
| Qwen2.5-VL-7B + ECRD<br>(Qwen2.5-VL-3B as visual decider) | 43.7 | 62.1       | 53.8 | 69.6 | 62.5 | 52.9 | 20.0      | 31.6 | 43.9                                                                            | 62.1 | 43.2 |
| Qwen2.5-VL-7B + ECRD                                      | 47.9 | 72.4       | 53.8 | 73.9 | 62.5 | 54.4 | 20.0      | 35.1 | 56.1                                                                            | 75.9 | 45.5 |

Table 3. TreeBench ablations on Qwen2.5-VL-7B. Best performances are highlighted in bold. The supervisor provides a stable boost and the visual decider adds the remaining lift; both components are necessary and complementary.

<span id="page-5-0"></span>

| Model                      | RH-Bench |       |        |  |  |  |
|----------------------------|----------|-------|--------|--|--|--|
|                            | Reas.    | Perc. | RH-AUC |  |  |  |
| Qwen2.5-VL-7B [1]          | 39.6     | 50.2  | 0.51   |  |  |  |
| Qwen2.5-VL-7B + supervisor | 42.0     | 53.3  | 0.54   |  |  |  |
| Qwen2.5-VL-7B + ECRD       | 46.4     | 57.1  | 0.58   |  |  |  |

Table 4. Results on RH-Bench. Reas. indicates *Reasoning*, and Perc. indicates *Perception*. Best performances are highlighted in bold. ECRD improves both Reasoning and Perception while increasing RH-AUC, indicating a better balance between reasoning and hallucination over longer chains.

## 3.4. Ablation study

The ablations on Qwen2.5-VL-7B in Tab. [3](#page-5-1) isolate where the lift comes from. First, GRIT-3B alone yields the lowest TreeBench accuracy among all rows, yet ECRD, invoking GRIT-3B only at uncertain steps, reaches 47.9% (+17.8 over GRIT-3B and +10.9 over the 7B base), which rules out the hypothesis that gains are driven by a stronger perception model and points instead to the decoding design, specifically ECRD's ability to fully leverage the small model exactly where it matters. Second, replacing VDGD's prefixwise minimum with our mean-over-prefix evidence scoring already helps without any decide, confirming that averaging stabilizes token selection. Third, adding a grounded decider matters: using Qwen2.5-VL-3B, a generic VLM, as the decider improves further, and swapping in the groundingoriented GRIT-3B yields the full ECRD gains, with the largest margins on visually consequential categories. The same staged pattern appears beyond TreeBench: as shown in Tabs. [4](#page-5-0) and [5,](#page-6-0) on RH-Bench and on the five general multimodal benchmarks, the variant without the decider consistently lies between the base and the full method, showing that both parts are necessary: the supervisor provides a robust, always-on stabilization under uncertainty, and the visual decider supplies sparse but decisive micro-observations only when ambiguity persists, converting the remaining hard steps and propagating benefits through rest of chain.

# 3.5. Efficiency analysis and uncertainty threshold

ECRD adds two sources of computation on top of the base decoder: (i) evidence scoring over the knee-selected candidate set and (ii) decider calls triggered when uncertainty persists. The former is lightweight. At step t, after knee truncation produces C<sup>t</sup> = {w1, . . . , wk}, scoring requires a pass over the evidence pool E<sup>t</sup> to form the KL-style preference s(w<sup>i</sup> |Et). Because each evidence contributes precomputed log-likelihoods on a separate and cache-disabled backend (stored as FP16 on CPU), the inference-time computation is O(k|Et|) with negligible GPU pressure. In practice k is single-digit and |Et| grows slowly, so this path contributes only a small fraction of total latency.

The heavier component is the visual decider. Let r denote the average number of decider calls per question. As Fig. [3](#page-6-1) shows, increasing the uncertainty threshold δ monotonically raises r (right panel), while accuracy (left panel) exhibits a consistent elbow: a rapid lift for small δ, followed by saturation. The knee for all five benchmarks concentrates around the gray dashed line (δ ≈ 0.08), which is precisely where the negotiated distribution most often flags genuinely ambiguous, visually charged steps. Benchmarks whose items hinge on precise perceptual grounding such as OCRBench and HallusionBench benefit early and strongly, their curves rise steeply with a small number of calls, whereas tasks that already align well with text-only priors like ChartQA show milder, earlier saturation. Math-Vista and V\* Bench lie in between: accuracy improves steadily as a handful of targeted micro-observations are injected, then flattens when further calls add little new information. Importantly, beyond the knee, the invocation of visual decider grows faster than accuracy, and some curves even level off or gently fluctuate, reflecting diminishing returns once the few decisive ambiguities have been resolved.

The cost trend aligns with a simple latency model grounded in Tab. [6.](#page-6-2) Let t<sup>0</sup> be the base time per question at δ = 0 (no calls) and l<sup>0</sup> be the average marginal latency of a single call. Then the end-to-end time obeys

$$T(\delta) \approx t_0 + l_0 r(\delta).$$
 (14)

<span id="page-6-3"></span><span id="page-6-0"></span>

| Model V* Bench             |             |         | ch      | MathVista | ChartQA | OCRBench | HallusionBench |
|----------------------------|-------------|---------|---------|-----------|---------|----------|----------------|
|                            | Attr        | Spatial | Overall |           |         |          |                |
| LLaVA-OneVision-7B [8]     | 73.0        | 60.5    | 68.1    | 63.2      | 80.0    | 62.2     | 55.1           |
| LLaVA-OneVision-7B + ECRD  | 74.8        | 65.8    | 71.2    | 67.9      | 86.3    | 73.8     | 63.7           |
| Qwen2.5-VL-7B [1]          | 73.9        | 67.1    | 71.2    | 68.2      | 84.4    | 82.3     | 61.3           |
| Qwen2.5-VL-7B + supervisor | 73.9        | 71.1    | 72.8    | 69.8      | 86.8    | 85.7     | 67.5           |
| Qwen2.5-VL-7B + ECRD       | <b>74.8</b> | 75.0    | 74.9    | 72.3      | 88.3    | 90.7     | 72.5           |

Table 5. Results on five general multimodal benchmarks. Better performances are highlighted in **bold**. ECRD brings consistent gains across diverse tasks on both Qwen2.5-VL-7B and LLaVA-OneVision-7B, demonstrating backbone-agnostic, task-general effectiveness.

<span id="page-6-1"></span>![](_page_6_Figure_2.jpeg)

Figure 3. Analysis of the uncertainty threshold  $\delta$ : accuracy as a function of  $\delta$  across five benchmarks, and the average visual decider invocation rate (calls per question) as  $\delta$  varies. The gray dashed line marks  $\delta = 0.08$ .

<span id="page-6-2"></span>

| Benchmark | V* Bench | MathVista | ChartQA | OCRBench | HallusionBench |
|-----------|----------|-----------|---------|----------|----------------|
| $t_0$     | 8.98     | 12.92     | 9.76    | 3.24     | 11.67          |
| $l_0$     | 1.32     | 1.46      | 1.30    | 1.12     | 1.43           |

Table 6. Average time per question at  $\delta=0$  ( $t_0$ , seconds per question) and global average latency of a single visual decider call ( $l_0$ , seconds). Note that  $l_0$  is computed as an overall mean across all benchmarks and  $\delta$  settings, and is not restricted to the  $\delta=0$  condition. All tests are conducted on a single H20-NVLink GPU.

with  $t_0$  varying by benchmark and  $l_0$  staying in a narrow band (single-second scale) across settings. Combined with Fig. 3, this explains the observed cost–accuracy balance: near  $\delta \approx 0.08$ , accuracy captures most of the attainable gain while r remains in the low single-digit regime, yielding modest overhead relative to  $t_0$ , whereas pushing  $\delta$  higher mainly increases r (and thus T) with little additional accuracy. Hence we adopt  $\delta = 0.08$  as a default: it lies at the elbow where added calls cease to be cost-effective, yet the model already recovers most of the gains in Tabs. 1 to 5.

## 3.6. Qualitative analysis: how ECRD works

As shown in Figs. 2 and 4, we present three representative cases that correct typical failure modes, corresponding to ECRD's two components—the supervisor and the visual decider: (i) negotiated reweighting with textual evidence and (ii) on-demand visual arbitration when uncertainty persists.

Case A: Negotiated evidence resolves the step. As shown in Fig. 4, the question asks whether rubber cars are fewer than brown jets. At critical steps, ECRD does not trig-

ger the decider because of the large reweighted gap, instead, it reweights the candidates so that tokens consistent with evidence are favored, yielding the correct answer "Yes".

Case B.1: Visual decider supplies mid-chain grounding. As shown in Fig. 2 (full examples are provided in supplementary material), at the key step the candidate set is { "blue", "red" }. The base slightly prefers "red", while the evidence-induced distribution prefers "blue"; the negotiated gap remains small, so the trigger fires. This choice is pivotal for the rest of the chain: if the model commits to "red", the subsequent localization and description would drift toward the red garment and propagate errors. The visual decider reads the image with the current prefix and returns the grounding sentence: "The first dress from the right-hand side is blue, partially hidden by the tree." ECRD forces "blue" for the current step and adds this sentence to the pool. Later tokens then follow this micro-observation to correctly describe all three dresses and reach the right final color judgment for the queried position.

Case B.2: Visual decider supplies the final answer. As shown in Fig. 4, here the candidate set at the decisive step is { "5", "3" }. The base leans toward "5", the evidence-induced distribution leans toward "3", and the small negotiated gap triggers the decider. The visual decider localizes the tag behind the box and returns: "The number behind the cardboard box with the 'favorita' brand and banana illustration is '300'." ECRD commits token "3" and inserts the sentence into the pool; at the next two steps the supervisor

<span id="page-7-0"></span>Question: Is the number of rubber cars less than the number of brown jets?

![](_page_7_Picture_1.jpeg)

#### Greedy Decoding

There are two rubber cars in the image: one cyan car and one green car. There are two brown jets in the image: one large jet and one small jet. Since the number of rubber cars (2) is equal to the number of brown jets (2). Therefore, the answer is No.

Question: Recognize the question and options in the image and answer it.

![](_page_7_Picture_5.jpeg)

## ECRD Case A: negotiated evidence resolves the step.

## Critical Evidence:

The image shows 9 objects in total with colors including yellow, blue, cyan, green, brown, gray, and orange. A yellow bicycle is on the left, a blue car is near the center, a cyan van is near the center, a yellow train is near the center, a gray train is in front, a brown road bike is on the right, a green car is on the right, a brown small airplane is near the centerright, and a brown large airplane is at the back. For the task, note that there are 2 brown jets (one large at the back and one small near the center-right), and there is 1 rubber car (a rubber cyan minivan) near the center. No text is visible.

## Model Answer:

Let's count the rubber cars and brown jets. There is 1 rubber car (a cyan minivan) near the center, and there are 2 brown jets (one large and one small) with one at the back and one near the center-right. Since the number of rubber cars (1) is less than the number of brown jets (2), the statement is true. Therefore, the answer is Yes.

## ECRD Case B: visual decider corrects the chain.

#### Model Answer Prefix:

The question asks for the object located directly behind the cardboard box with the \"favorita\" brand and banana illustration. In the image, there is a cardboard box with the \"favorita\" logo and bananas on it. Directly behind this box, there is a price tag with the number \"

Candidate Set: { "5", "3" }

## Negotiation (token, probability):

base model: (5, 0.498) (3, 0.483) evidence: (5, 0.503) (3, 0.478) reweighted: (5, 0.501) (3, 0.480)

## Visual Decider Output:

coordinates: see the red box in the left image. evidence: The number behind the cardboard box with the \"favorita\" brand and banana illustration is \"300\".

## Model Final Answer:

The question asks for the object located directly behind the cardboard box with the \"favorita\" brand and banana illustration. In the image, there is a cardboard box with the \"favorita\" logo and bananas on it. Behind this box, there is a price tag with the number \"300\" written on it. Therefore, the correct answer should be the price tag with \"300\" in text.

![](_page_7_Figure_21.jpeg)

<span id="page-7-1"></span>![](_page_7_Figure_22.jpeg)

Figure 5. Breakdown of ECRD's gain on TreeBench based on Qwen2.5-VL-7B.

prefers tokens consistent with this evidence, selecting "0" then "0" without additional calls. The chain thus outputs the correct answer "300".

Where the gains come from. Based on a detailed statistical analysis of the experimental results, we find that on Qwen2.5-VL-7B, ECRD yields a +10.9-point overall improvement on TreeBench. As shown in Fig. 5, within this

lift, cases where the visual decider directly outputs the final answer account for 11.4% of the gain, while cases where the decider injects a mid-chain visual grounding that unlocks the rest of the reasoning account for 18.2%. The remaining improvement primarily comes from the supervisor's negotiated reweighting and the indirect benefits of an expanding evidence pool, which stabilizes later token choices even when no further decider calls are needed.

## 4. Conclusion

We presented ECRD, a decoding-time, training-free, plugand-play framework for visually grounded reasoning. By reconciling base probabilities with an evidence-guided distribution, ECRD preserves model confidence, invokes a lightweight visual decider only when necessary, and propagates visual grounding through reasoning chain. Experiments show consistent gains across diverse benchmarks.

# Acknowledgements

This work was supported in part by the National Natural Science Foundation of China under Grant T2225025 and 62371121, in part by the National Key Research and Development Program of China under Grant 2024YFF1206700, and in part by the Fundamental Research Funds for the Central Universities. AIIA refers to the Key Laboratory of New Generation Artificial Intelligence Technology and Its Interdisciplinary Applications (Southeast University).

# References

- <span id="page-8-13"></span>[1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025. [4,](#page-3-1) [5,](#page-4-2) [6,](#page-5-2) [7](#page-6-3)
- <span id="page-8-22"></span>[2] Google DeepMind. Gemini-2.5-flash, 2025. [4,](#page-3-1) [5](#page-4-2)
- <span id="page-8-23"></span>[3] Google DeepMind. Gemini-2.5-pro, 2025. [4,](#page-3-1) [5](#page-4-2)
- <span id="page-8-12"></span>[4] Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze Guan, and Xin Eric Wang. Grit: Teaching mllms to think with images. *arXiv preprint arXiv:2505.15879*, 2025. [4,](#page-3-1) [6](#page-5-2)
- <span id="page-8-11"></span>[5] Sreyan Ghosh, Chandra Kiran Reddy Evuru, Sonal Kumar, Utkarsh Tyagi, Oriol Nieto, Zeyu Jin, and Dinesh Manocha. Visual description grounding reduces hallucinations and boosts reasoning in lvlms. *arXiv preprint arXiv:2405.15683*, 2024. [3](#page-2-5)
- <span id="page-8-18"></span>[6] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14375–14385, 2024. [4](#page-3-1)
- <span id="page-8-0"></span>[7] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. *arXiv preprint arXiv:2505.07062*, 2025. [1](#page-0-1)
- <span id="page-8-19"></span>[8] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. *arXiv preprint arXiv:2408.03326*, 2024. [4,](#page-3-1) [5,](#page-4-2) [7](#page-6-3)
- <span id="page-8-6"></span>[9] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In *International conference on machine learning*, pages 19730– 19742. PMLR, 2023. [2](#page-1-0)
- <span id="page-8-5"></span>[10] Chengzhi Liu, Zhongxing Xu, Qingyue Wei, Juncheng Wu, James Zou, Xin Eric Wang, Yuyin Zhou, and Sheng Liu. More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models, 2025. [1,](#page-0-1) [4](#page-3-1)
- <span id="page-8-7"></span>[11] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *Advances in neural information processing systems*, 36:34892–34916, 2023. [2](#page-1-0)
- <span id="page-8-17"></span>[12] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen

- Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. *Science China Information Sciences*, 67(12):220102, 2024. [4](#page-3-1)
- <span id="page-8-15"></span>[13] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. *arXiv preprint arXiv:2310.02255*, 2023. [4](#page-3-1)
- <span id="page-8-16"></span>[14] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. *arXiv preprint arXiv:2203.10244*, 2022. [4](#page-3-1)
- <span id="page-8-20"></span>[15] OpenAI. Openai-gpt-4o, 2024. [4,](#page-3-1) [5](#page-4-2)
- <span id="page-8-21"></span>[16] OpenAI. Openai-o3, 2025. [4,](#page-3-1) [5](#page-4-2)
- <span id="page-8-8"></span>[17] Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. *arXiv preprint arXiv:1809.02156*, 2018. [2](#page-1-0)
- <span id="page-8-1"></span>[18] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. *Advances in Neural Information Processing Systems*, 37:8612–8642, 2024. [1](#page-0-1)
- <span id="page-8-10"></span>[19] Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. *arXiv preprint arXiv:2505.15966*, 2025. [2,](#page-1-0) [4,](#page-3-1) [5](#page-4-2)
- <span id="page-8-2"></span>[20] Guangyan Sun, Mingyu Jin, Zhenting Wang, Cheng-Long Wang, Siqi Ma, Qifan Wang, Tong Geng, Ying Nian Wu, Yongfeng Zhang, and Dongfang Liu. Visual agents as fast and slow thinkers. *arXiv preprint arXiv:2408.08862*, 2024. [1](#page-0-1)
- <span id="page-8-24"></span>[21] D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 11888–11898, 2023. [5](#page-4-2)
- <span id="page-8-3"></span>[22] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1.5: Scaling reinforcement learning with llms. *arXiv preprint arXiv:2501.12599*, 2025. [1](#page-0-1)
- <span id="page-8-4"></span>[23] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. *arXiv preprint arXiv:2504.07491*, 2025. [1](#page-0-1)
- <span id="page-8-14"></span>[24] Haochen Wang, Xiangtai Li, Zilong Huang, Anran Wang, Jiacong Wang, Tao Zhang, Jiani Zheng, Sule Bai, Zijian Kang, Jiashi Feng, et al. Traceable evidence enhanced visual grounded reasoning: Evaluation and methodology. *arXiv preprint arXiv:2507.07999*, 2025. [4,](#page-3-1) [5](#page-4-2)
- <span id="page-8-9"></span>[25] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in neural information processing systems*, 35:24824–24837, 2022. [2](#page-1-0)
- <span id="page-8-25"></span>[26] Mingrui Wu, Xinyue Cai, Jiayi Ji, Jiale Li, Oucheng Huang, Gen Luo, Hao Fei, Guannan Jiang, Xiaoshuai Sun, and Rongrong Ji. Controlmllm: Training-free visual prompt learning

- for multimodal large language models. *Advances in Neural Information Processing Systems*, 37:45206–45234, 2024. [5](#page-4-2)
- <span id="page-9-1"></span>[27] Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal llms. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13084–13094, 2024. [4](#page-3-1)
- <span id="page-9-3"></span>[28] Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, and Enhong Chen. Woodpecker: Hallucination correction for multimodal large language models. *Science China Information Sciences*, 67(12):220105, 2024. [5](#page-4-2)
- <span id="page-9-0"></span>[29] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. *arXiv preprint arXiv:2505.14362*, 2025. [2,](#page-1-0) [4,](#page-3-1) [5](#page-4-2)
- <span id="page-9-2"></span>[30] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. *arXiv preprint arXiv:2504.10479*, 2025. [4,](#page-3-1) [5](#page-4-2)