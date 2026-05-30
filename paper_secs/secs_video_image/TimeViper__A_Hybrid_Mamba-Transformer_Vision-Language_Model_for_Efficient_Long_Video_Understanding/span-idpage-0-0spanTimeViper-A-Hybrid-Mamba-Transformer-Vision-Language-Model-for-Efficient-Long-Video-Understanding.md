# <span id="page-0-0"></span>TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding

Boshen Xu<sup>1</sup>∗‡ Zihan Xiao<sup>1</sup><sup>∗</sup> Jiaze Li<sup>2</sup> Jianzhong Ju<sup>2</sup> Zhenbo Luo<sup>2</sup> Jian Luan<sup>2</sup> Qin Jin<sup>1</sup>† <sup>1</sup> AIM3 Lab, Renmin University of China <sup>2</sup> MiLM Plus, Xiaomi Inc.

Project Page: <https://xuboshen.github.io/TimeViper/>

![](_page_0_Figure_4.jpeg)

Figure 1. We present TimeViper, a hybrid Mamba-Transformer vision-language model for efficient long video understanding. We reveal the severe vision token redundancy and a vision-to-text information aggregation phenomenon in hybrid models. To this end, we introduce TransV, the first token-transfer module that compresses vision tokens into text tokens inside the LLM, enabling the model to process over 10,000 frames. Benefitting from the Mamba layers' O(n) computation and O(1) cache cost, TimeViper generates 40.1% more tokens per second than Qwen3 [\[97\]](#page-17-0) when processing 32k input tokens (approximately 2k frames at 16 tokens per frame) and producing 1k output tokens with batch size 32. TimeViper delivers performance competitive with Transformer-based MLLMs on public benchmarks, including multi-choice QA on VideoMME [\[29\]](#page-13-0) (vs. Video-XL [\[73\]](#page-16-0)), temporal video grounding on Charades [\[74\]](#page-16-1) (vs. VTimeLLM [\[36\]](#page-13-1)), video detailed captioning on VDC [\[14\]](#page-12-0) (vs. AuroraCap [\[14\]](#page-12-0)), and hour-long video understanding on LVBench [\[85\]](#page-16-2) (vs. Gemini-1.5-Pro [\[80\]](#page-16-3)).

