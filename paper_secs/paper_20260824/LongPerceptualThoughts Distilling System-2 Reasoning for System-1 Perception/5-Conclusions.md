# **5 Conclusions**

In this work, we explore how system-2 reasoning can enhance vision-centric tasks. We introduce a novel, scalable data synthesis framework that requires only dense image captions. The framework generates verifiable multiple-choice questions, extracts simple chains of thought (CoTs) from vision-language models (VLMs), and expands them into rich, longform reasoning traces using frontier reasoning models. This process yields LongPerceptualThoughts, a synthetic dataset containing 30k detailed reasoning traces for perceptual tasks. Fine-tuning Qwen2.5-VL-7B-Instruct on LongPerceptualThoughts improves performance by +3.4 points across five vision benchmarks, including ann +11.8-point gain on V<sup>∗</sup> Bench. Remarkably, despite being trained on vision tasks, the model also improves by +2 points on the out-of-distribution text-only reasoning benchmark MMLU-Pro.

### **Acknowledgements**

We thank Rafid Mahmood, Jaehun Jung, Jen-Hao Cheng, Ali Hatamizadeh, Ximing Lu, Hyunwoo Kim and Amlan Kar for their helpful comments and feedback on an early discussions and draft of this paper.

