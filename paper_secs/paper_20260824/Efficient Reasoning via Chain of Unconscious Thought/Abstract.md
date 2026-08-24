# Abstract

Large Reasoning Models (LRMs) achieve promising performance but compromise token efficiency due to verbose reasoning processes. Unconscious Thought Theory (UTT) posits that complex problems can be solved more efficiently through internalized cognitive processes. Inspired by UTT, we propose a new reasoning paradigm, termed Chain of Unconscious Thought (CoUT), to improve the token efficiency of LRMs by guiding them to mimic human unconscious thought and internalize reasoning processes. Concretely, we first prompt the model to internalize the reasoning by thinking in the hidden layer. Then, we design a bag of token-efficient strategies to further help models reduce unnecessary tokens yet preserve the performance. Our work reveals that models may possess beneficial unconscious thought, enabling improved efficiency without sacrificing performance. Extensive experiments demonstrate the effectiveness of CoUT. Remarkably, it surpasses CoT by reducing token usage by 47.62% while maintaining comparable accuracy, as shown in Figure [1.](#page-0-0) The code of CoUT is available at this link[1](#page-0-1) .

<span id="page-0-0"></span>> **[图片提取文字 (无描述)]:**
> Accuracy (%) Token Count 100 800 676.85 700 95 600 91.19 534.24 90 88.50 88.40 500 445.91 85 400 354.46 80.89 300 80 200 75 100 70 CoUT CoT CoD **CCoT**
![](_page_0_Figure_6.jpeg)

Figure 1: Average Performance and Tokens of CoUT and Baselines for 4 LRMs over 4 Benchmarks.

