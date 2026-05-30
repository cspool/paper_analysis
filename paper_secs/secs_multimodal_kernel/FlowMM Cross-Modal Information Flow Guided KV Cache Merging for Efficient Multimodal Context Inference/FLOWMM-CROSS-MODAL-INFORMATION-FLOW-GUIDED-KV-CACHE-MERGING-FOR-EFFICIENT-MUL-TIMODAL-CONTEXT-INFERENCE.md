# FLOWMM: CROSS-MODAL INFORMATION FLOW GUIDED KV CACHE MERGING FOR EFFICIENT MUL-TIMODAL CONTEXT INFERENCE

Kunxi Li<sup>1</sup> , Yufan Xiong<sup>2</sup> , Zhonghua Jiang<sup>1</sup> , Yiyun Zhou<sup>1</sup> , Zhaode Wang<sup>3</sup> , Chengfei Lv<sup>3</sup> , Shengyu Zhang1<sup>∗</sup>

<sup>1</sup>Zhejiang University, <sup>2</sup>Huazhong Agricultural University, <sup>3</sup>Alibaba kunxili@zju.edu.cn, sy zhang@zju.edu.cn

### ABSTRACT

Traditional KV cache eviction strategies, which discard less critical KV-pairs based on attention scores, often degrade generation quality, causing context loss or hallucinations. Recent efforts shift toward KV merging, merging eviction tokens with retention tokens based on similarity. However, in multimodal scenarios, distributional biases across modality tokens and attentional biases in cross-modal interactions limit its effectiveness. This work introduces FlowMM, an adaptive framework for cross-modal information flow-guided multimodal KV cache merging. FlowMM leverages cross-modal information flow to dynamically apply layerspecific merging strategies, capturing modality-specific patterns while preserving contextual integrity. Furthermore, we introduce a sensitivity-adaptive token matching mechanism that jointly evaluates token similarity and task-critical sensitivity, merging low-risk tokens while safeguarding high-sensitivity ones. Extensive experiments across diverse leading MLLMs show that FlowMM reduces KV cache memory by 80% to 95% and decoding latency by 1.3-1.8×, while maintaining competitive task performance.

