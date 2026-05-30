# Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction

Zhenmei Shi<sup>∗</sup> Yifei Ming† Xuan-Phi Nguyen‡ Yingyu Liang§ Shafiq Joty¶

#### Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities in handling long context inputs, but this comes at the cost of increased computational resources and latency. Our research introduces a novel approach for the long context bottleneck to accelerate LLM inference and reduce GPU memory consumption. Our research demonstrates that LLMs can identify relevant tokens in the early layers before generating answers to a query. Leveraging this insight, we propose an algorithm that uses early layers of an LLM as filters to select and compress input tokens, significantly reducing the context length for subsequent processing. Our method, Gem-Filter, demonstrates substantial improvements in both speed and memory efficiency compared to existing techniques, such as standard attention and SnapKV/H2O. Notably, it achieves a 2.4× speedup and 30% reduction in GPU memory usage compared to SOTA methods. Evaluation on the Needle in a Haystack task shows that GemFilter significantly outperforms standard attention, SnapKV and demonstrates comparable performance on the LongBench challenge. GemFilter is simple, training-free, and broadly applicable across different LLMs. Crucially, it provides interpretability by allowing humans to inspect the selected input sequence. These findings not only offer practical benefits for LLM deployment, but also enhance our understanding of LLM internal mechanisms, paving the way for further optimizations in LLM design and inference. Our code is available at <https://github.com/SalesforceAIResearch/GemFilter>.

<sup>∗</sup> zhmeishi@cs.wisc.edu. University of Wisconsin-Madison.

<sup>†</sup> yifei.ming@salesforce.com. Salesforce AI Research.

xnguyen@salesforce.com. Salesforce AI Research.

yingyul@hku.hk. The University of Hong Kong. yliang@cs.wisc.edu. University of Wisconsin-Madison.

<sup>¶</sup> sjoty@salesforce.com. Salesforce AI Research.

