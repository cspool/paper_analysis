# Beyond Accuracy: Evaluating Grounded Visual Evidence in Thinking with Images

**Xuchen Li**<sup>1\*</sup> **Xuzhao Li**<sup>2\*</sup> **Renjie Pi**<sup>3†</sup> **Shiyu Hu**<sup>2</sup> **Jian Zhao**<sup>1,5</sup> **Jiahui Gao**<sup>4‡</sup> <sup>1</sup>ZGCA, <sup>2</sup>NTU, <sup>3</sup>HKUST, <sup>4</sup>HKU, <sup>5</sup>ZGCI

xuzhaoli2001@gmail.com, xuchenli1030@gmail.com, rpi@connect.ust.hk, ggaojiahui@gmail.com

#### Abstract

Despite the remarkable progress of Vision-Language Models (VLMs) in adopting "Thinking-with-Images" capabilities, accurately evaluating the authenticity of their reasoning process remains a critical challenge. Existing benchmarks mainly rely on outcomeoriented accuracy, lacking the capability to assess whether models can accurately leverage fine-grained visual cues for multi-step reasoning. To address these limitations, we propose ViEBench, a process-verifiable benchmark designed to evaluate faithful visual reasoning. Comprising 200 multi-scenario high-resolution images with expert-annotated visual evidence, ViEBench uniquely categorizes tasks by difficulty into perception and reasoning dimensions, where reasoning tasks require utilizing localized visual details with prior knowledge. To establish comprehensive evaluation criteria, we introduce a dual-axis matrix that provides fine-grained metrics through four diagnostic quadrants, enabling transparent diagnosis of model behavior across varying task complexities. Our experiments yield several interesting observations: (1) VLMs can sometimes produce correct final answers despite grounding on irrelevant regions, and (2) they may successfully locate the correct evidence but still fail to utilize it to reach accurate conclusions. Our findings demonstrate that ViEBench can serve as a more explainable and practical benchmark for comprehensively evaluating the effectiveness agentic VLMs. The codes will be released at: https://github.com/Xuchen-Li/ViEBench.

#### 1 Introduction

Recent advancements in Vision-Language Models (VLMs) (Bai et al., 2025a; Su et al., 2025a; Lai et al., 2025; Team et al., 2025b,a; MiniMax,

<span id="page-0-0"></span>Table 1: Comparison of ViEBench with existing Multimodal Benchmarks. ViEBench is uniquely categorized into perception and reasoning tasks, and provides expertannotated bounding box (BBox) for visual evidence, enabling a precise process evaluation.

| Bench       | #QA Pairs | Percept      | Reason | BBox     | Process<br>Evaluation |
|-------------|-----------|--------------|--------|----------|-----------------------|
| V* Bench    | 191       | ✓            | ×      | ×        | ×                     |
| HRBench     | 1600      | $\checkmark$ | ×      | ×        | ×                     |
| InfoVQA     | 2801      | $\checkmark$ | ×      | ×        | ×                     |
| VisualProbe | 515       | $\checkmark$ | ×      | ×        | ×                     |
| ViEBench    | 200       | ✓            | ✓      | <b>√</b> | ✓                     |

2025) have entered in a new era of "Thinking-with-Images," where models no longer perceive images as static inputs but instead actively interact with them through dynamic visual operations (Zheng et al., 2025). By adopting agentic behaviors such as autonomous zooming, state-of-the-art agentic models like the o3 (OpenAI, 2025) and Qwen3-VL (Bai et al., 2025a) have demonstrated an unprecedented ability to resolve fine-grained details within high-resolution scenes. This shift from passive perception to active visual reasoning has enabled VLMs to tackle complex tasks in real-world scenarios, ranging from industrial inspection to urban navigation (Su et al., 2025b; Zhang et al., 2023; Li et al., 2025c; Cao et al., 2025).

However, as VLMs gain the autonomy to manipulate their own visual input, a critical evaluation gap has emerged. As shown in Tab. 1, existing benchmarks (Wang et al., 2024; Wu and Xie, 2024; Lai et al., 2025; Mathew et al., 2022; Chen et al., 2024) are constrained by two fundamental limitations. First, their perception-oriented tasks can be addressed through fine-grained recognition alone, making them insufficient for evaluating tool-use capabilities in reasoning-intensive scenarios where models must integrate localized visual cues with prior knowledge. Second, these benchmarks rely solely on outcome-oriented metrics (Li et al., 2025d,e), treating models as "black

<sup>\*</sup>Equal contribution.

<sup>&</sup>lt;sup>†</sup>Project Leader.

<sup>&</sup>lt;sup>‡</sup>Corresponding Author.

boxes" and providing no diagnostic granularity to distinguish whether performance degradation results from grounding failures or from an inability to synthesize visual evidence into logical reasoning. Consequently, without fine-grained metrics to verify if a model's success relies on faithful reasoning or mere textual priors, it remains impossible to pinpoint specific weaknesses or guide targeted model improvements.

To bridge this gap, we introduce ViEBench, a novel diagnostic benchmark to evaluate the "Thinking-with-Images" capabilities of VLMs. First, we design ViEBench-R, a reasoning task that requires models to localize fine-grained visual cues, integrate them with prior knowledge, and perform multi-step logical reasoning. We also provide ViEBench-P, a perception task for comparable analysis. To ensure task quality and diversity, we curate 200 high-resolution images across four critical real-world scenarios: retail, urban, industry, and daily life. Crucially, we enforce extreme spatial sparsity, where critical visual evidence occupies less than 0.7% of the total image area on average. This strategic design ensures that essential visual cues remain sub-perceptual in global views, forcing models to execute precise local zooming operations.

Second, we introduce a dual-axis capability matrix based on the intersection over area (IoA) metric [\(Xiang et al.,](#page-9-9) [2023\)](#page-9-9). By comparing modelgenerated visual crops against expert-annotated gold BBox, we construct a grounding axis that quantifies localization accuracy, which is then crossed with the answer correctness axis to form four diagnostic quadrants (shown in Fig. [1\)](#page-2-0): *Valid Grounded Reasoning*, *Ground-Success Answer-Failure*, *Ungrounded Correct Answer*, and *Dual Ground-Answer Failure*. This decomposition enables us to evaluate whether performance degradation stems from grounding failures or reasoning deficiencies—a diagnostic capability completely absent in traditional accuracy-only metrics.

Our extensive evaluation of both end-to-end and agentic VLMs reveals several counter-intuitive findings. We identify a prevalent ungrounded correct answer phenomenon among agentic VLMs, suggesting that current benchmarks significantly overestimate model reliability. Furthermore, we uncover a ground-success and answer-failure bottleneck, where models successfully locate the required evidence but fail to synthesize it into a correct reasoning chain. These insights underscore

that the next frontier for VLMs lies not just in "where to look," but in how to deeply integrate cropped visual information into the reasoning chain. By providing a rigorous and transparent evaluation protocol, ViEBench serves as both a valuable benchmark for current models and a roadmap for the development of more robust and truly visual "thinking" agentic models.

This work makes three key contributions: 1) We introduce ViEBench across diverse real-world scenarios, uniquely featuring reasoning-oriented tasks with extreme spatial sparsity (avg. area < 0.7%), necessitating precise visual operations and faithful visual thinking. 2) We propose a process-verifiable evaluation paradigm that shifts from outcome-oriented metrics to a dual-axis capability matrix, enabling fine-grained analysis to differentiate grounding failures from reasoning deficiencies. 3) Our extensive evaluation of stateof-the-art agentic VLMs reveals several counterintuitive failure modes, offering actionable insights for developing more robust agentic vision models.

