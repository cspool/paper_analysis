# 5 Conclusion

We propose APB, a distributed long-context inference framework based on approximate attention, achieving speedups up to 9.2×, 4.2×, and 1.6× compared to FLASHATTN, RINGATTN, and STARATTN, respectively, without observable performance degradation. APB constructs the passing blocks with minimal communication cost by sending only core KV pairs among hosts. The experimental results on ∞Bench and RULER, using various models and sequence lengths, demonstrate that APB delivers faster inference speeds while maintaining or exceeding performance. Furthermore, APB is adaptable to diverse distribution configurations and models of varying sizes. Next, we aim to accelerate the decoding process in APB, where the KV cache is distributed across different hosts.

<span id="page-8-1"></span><sup>2</sup> https://github.com/huggingface/transformers

<span id="page-8-2"></span><sup>3</sup> https://github.com/mobiusml/hqq

<span id="page-8-4"></span><sup>4</sup> https://huggingface.co/datasets/wenbopan/anti-haystack

