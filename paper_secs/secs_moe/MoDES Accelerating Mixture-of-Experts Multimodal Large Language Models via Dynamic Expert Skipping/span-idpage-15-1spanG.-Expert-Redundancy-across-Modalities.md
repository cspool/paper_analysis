# <span id="page-15-1"></span>**G.** Expert Redundancy across Modalities

<span id="page-15-2"></span>![](_page_15_Figure_2.jpeg)

Figure III. Task performance across various numbers of top-k routed experts applied to tokens of different modalities for Kimi-VL-A3B-Instruct [50].

In this section, we analyze expert redundancy across modalities. As shown in Fig. III, reducing k for vision tokens causes task performance to drop more slowly than for text tokens. This indicates greater redundancy among experts for vision tokens, allowing more aggressive skipping than for text tokens. It also motivates modality-aware strategies for *expert skipping*.