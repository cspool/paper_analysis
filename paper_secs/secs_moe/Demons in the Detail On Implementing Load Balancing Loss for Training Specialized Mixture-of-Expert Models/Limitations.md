# **Limitations**

This paper primarily focuses on analyzing the impact of micro-batch LBL on LLMs during the pre-training stage. It does not further investigate its effects during fine-tuning or in the vision and multi-modality domains. Our analysis of specialization is mainly centred on the selection frequency across different domains without conducting more rigorous validation. Relaxing micro-batch LBL can introduce some latency. Future work could consider including more diverse sequences within each micro-batch to mitigate this local imbalance issue.

