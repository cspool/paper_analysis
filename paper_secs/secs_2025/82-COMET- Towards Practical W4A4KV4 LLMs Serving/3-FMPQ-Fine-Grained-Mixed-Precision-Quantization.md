# 3 FMPQ: Fine-Grained Mixed-Precision Quantization

To address the challenges of activation quantization in posttraining quantization, this section first analyzes the characteristics of activation distributions in LLMs and then proposes a mixed-precision quantization algorithm to achieve low-bit quantization of LLM activations. The proposed FMPQ effectively reduces the computational and storage costs of LLM inference, serving as the foundational enabler of our COMET inference framework.

