# StreamingEval: A Unified Evaluation Protocol towards Realistic Streaming Video Understanding

Guowei Tang<sup>1</sup>, Tianwen Qian<sup>1†</sup>, Huanran Zheng<sup>1</sup>, Yifei Wang<sup>1</sup>, Xiaoling Wang<sup>1</sup>

<sup>1</sup>East China Normal University, Shanghai, China

51274404096@stu.ecnu.edu.cn, twqian@cs.ecnu.edu.cn

#### **Abstract**

Real-time, continuous understanding of visual signals is essential for real-world interactive AI applications, and poses a fundamental system-level challenge. Existing research on streaming video understanding, however, typically focuses on isolated aspects such as question-answering accuracy under limited visual context or improvements in encoding efficiency, while largely overlooking practical deployability under realistic resource constraints. To bridge this gap, we introduce StreamingEval, a unified evaluation framework for assessing the streaming video understanding capabilities of Video-LLMs under realistic constraints. StreamingEval benchmarks both mainstream offline models and recent online video models under a standardized protocol, explicitly characterizing the trade-off between efficiency, storage and accuracy. Specifically, we adopt a fixed-capacity memory bank to normalize accessible historical visual context, and jointly evaluate visual encoding efficiency, text decoding latency, and task performance to quantify overall system deployability. Extensive experiments across multiple datasets reveal substantial gaps between current Video-LLMs and the requirements of realistic streaming applications, providing a systematic basis for future research in this direction. Codes will be released at https://github. com/wwgTang-111/StreamingEval1.

#### 1 Introduction

As video large language models (Video-LLMs) (Maaz et al., 2024; Zhang et al., 2023; Song et al., 2024; Jin et al., 2024) continue to advance, applications such as embodied robots (Driess et al., 2023; Brohan et al., 2023; Wang et al., 2026), live-streaming assistants (Chen et al., 2024; Xu et al., 2025), and autonomous driving systems (Levinson

<span id="page-0-0"></span>![](_page_0_Picture_10.jpeg)

Figure 1: Illustration of the conventional offline video understanding paradigm versus the streaming paradigm. Top: offline inference with full access to the video. Middle: pseudo-streaming inference, which truncates videos at query timestamps but still processes each clip in an offline manner. Bottom: realistic streaming inference with real-time incremental input and limited memory bank.

et al., 2011; Qian et al., 2024; Brödermann et al., 2025) are becoming increasingly feasible. In these scenarios, visual inputs arrive continuously in an incremental manner, requiring models to process them in real time under strict latency and resource constraints. However, existing mainstream Video-LLMs are designed and evaluated in offline settings, where the input videos are pre-recorded and fully accessible by the model. In contrast, streaming video understanding requires the model to process continuously arriving inputs without access to future frames, while sustaining instant responsiveness over potentially unbounded time horizons.

This fundamental mismatch between offline and streaming settings gives rise to several technical challenges that are largely absent in conventional video understanding tasks. First, streaming mod-

<sup>&</sup>lt;sup>†</sup>Corresponding author.

els must reason with incomplete and evolving visual context, as only past and current frames are available at any time. Second, streaming video is characterized by unbounded temporal duration, and long-running applications require persistent visual memory, but limited GPU resources inevitably lead to out-of-memory failures. Third, streaming systems impose stringent efficiency requirements. If visual encoding or memory updates fall behind the incoming frame rate, frames will accumulate and break real-time responsiveness. Similarly, excessive decoding latency directly degrades the interactive experience. Figure [1](#page-0-0) illustrates the differences between offline and streaming video understanding paradigms.

Recent studies have begun to investigate individual aspects of these challenges in isolation. For example, OVO-Bench [\(Li et al.,](#page-9-5) [2025\)](#page-9-5) introduces an evaluation benchmark aligned with streaming context constraints and analyzes the model performance across different timestamps of questions (Backward Tracing, Real-Time Visual Perception, and Forward Active Responding). Several recent studies investigate the impact of visual encoding efficiency on online video processing [\(Yao et al.,](#page-10-4) [2025;](#page-10-4) [Wang et al.,](#page-10-5) [2025c;](#page-10-5) [Zeng et al.,](#page-10-6) [2025;](#page-10-6) [Chen](#page-8-3) [et al.,](#page-8-3) [2025;](#page-8-3) [Ning et al.,](#page-9-6) [2025\)](#page-9-6), or quantify the effects of text decoding latency [\(Reddi et al.,](#page-9-7) [2019\)](#page-9-7). However, the diversity of evaluation methodologies has so far prevented the emergence of a unified and systematic evaluation framework. In particular, some benchmarks rely on pseudo-streaming settings, where videos are truncated at the query timestamp but still processed offline, limiting fair, reproducible, and deployment-oriented comparisons across models.

To bridge this gap, we introduce StreamingEval, a unified evaluation framework for video understanding under realistic streaming constraints. StreamingEval formalizes streaming video understanding as a system-level problem and provides a standardized evaluation protocol with metrics that jointly characterize accuracy, latency, and memory usage. Concretely, this framework implements a standardized streaming pipeline with three modules: (i) a frame player that continuously emits raw video frames at a fixed frame rate; (ii) an encodingand-memory processor that performs per-frame visual encoding and memory updates according to each model's design; (iii) and a response generator that, upon receiving a query, encodes the query, loads the current visual memory, and invokes the

language model to generate answers. Based on this framework, we evaluate 12 representative online and offline Video-LLMs. Native online models retain their original streaming mechanisms and configurations, while offline models are modified with a fixed-capacity visual memory bank managed via a first-in-first-out (FIFO) policy, standardizing the accessible historical context during streaming inference.

Extensive experiments across multiple datasets demonstrate that current methods claimed to be "online" fail to operate reliably under strict streaming constraints. Moreover, under identical streaming settings, mainstream offline models often outperform specialized online models at the cost of higher resource consumption. Performance can be improved by enlarging memory capacity or raising visual input resolution, but this entails a trade-off with efficiency and deployability. Overall, our results reveal substantial gaps between current Video-LLMs and the requirements of realistic streaming applications, highlighting open challenges for future research. In summary, our main contributions are as follows:

- We propose StreamingEval, a framework for assessing both the holistic capabilities of models and their deployability in streaming settings.
- We establish a scalable suite of online evaluation metrics and a unified protocol, enabling fair comparisons across models under consistent constraints.
- We conduct systematic empirical evaluations and analyses of representative state-of-theart online and offline models, providing a reusable benchmark and clear directions for future research.

