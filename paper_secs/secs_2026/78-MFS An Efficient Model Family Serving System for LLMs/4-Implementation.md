# 4 Implementation

In our implementation, we applied the concept of "knowledge precipitation" to restructure a pre-trained LLM into a multi-level model. This involved full-parameter fine-tuning, which allowed each layer of the original model to condense its functionalities into discrete submodels, each capable of standalone operation. This hierarchical structuring not only preserves the integrity and depth of the learned representations but also facilitates more granular control over the inference process.

Training infrastructure. We use two servers, each equipped with eight NVIDIA H800 SXM5 GPUs. Each server has 2×56 Intel Xeon Gold CPU cores, and 2TB memory. Two servers are interconnected with eight 400Gbps NDR InfiniBand network interfaces.

Models and datasets. We chose the Llama2 family, a widelyused series of large models that includes three official versions: Llama2-7B, Llama2-13B, and Llama2-70B. Due to computational resource constraints, i.e., 16× NVIDIA H800 GPUs can only support up to Llama2-13B, we only fine-tune Llama2- 7B and Llama2-13B (the chat version). To test the generality of our approach, we also evaluate the Qwen model family, which includes Qwen-4B and Qwen-7B. To demonstrate the resilience and time efficiency of knowledge precipitation, we use a small dataset [\[4\]](#page-12-2) with 9.85k dialogues on Hugging Face for fine-tuning.

Fine-tuning parameters. The model fine-tuning mostly follows the fine-tuning parameters in the LLAMA-2 paper. We use the AdamW optimizer, with a learning rate of 2e-5. The half-periodic cosine learning rate function is used. The weight decay amplitude is 0.1, and we apply gradient clipping of 0.3. With 8 gradient accumulations, the equivalent batch size of training is 64. The input sequence length is 4096 tokens. The model is fine-tuned for one epoch on a generation/dialogue mixed dataset with about 0.66T (665.3 billion) tokens, a total of 2500 iterations, and a total time of about 24 hours.

For inference serving, we have two settings, one is a 2× NVIDIA A100 GPUs server with 2 × 48 Intel Xeon Gold CPU cores and 512G memory. The second is an 8× NVIDIA 3090 GPUs server with 80 Intel Xeon Gold CPU cores and 256G memory.

