# <span id="page-14-2"></span>**D. Tools in training**

Below is the complete list of tools used in the training. For each example rollout, we randomly sample a subset of them to simulate heterogeneous availability of tools:

- Query writer: GPT-5 [\[23\]](#page-11-4), GPT-5-mini [\[23\]](#page-11-4), meta-llama/Llama-3.3-70B-Instruct [\[26\]](#page-11-7), meta-llama/Llama-3.1-8B-Instruct [\[26\]](#page-11-7), deepseek-ai/DeepSeek-R1 [\[57\]](#page-12-16), nvidia/Llama-3\_1-Nemotron-Ultra-253B-v1 [\[29\]](#page-11-10), microsoft/Phi-4-mini-instruct [\[58\]](#page-12-17), google/gemma-3-27b-it [\[33\]](#page-11-14), Qwen/Qwen3-32B [\[27\]](#page-11-8)
- Web search: We use Tavily search API [4](#page-14-4) to provide orchestrator real-time web access.
- Local search: Faiss index with Qwen/Qwen3-Embedding-8B [\[22\]](#page-11-3)

<span id="page-14-4"></span><sup>4</sup>[https://www](https://www.tavily.com/)*.*tavily*.*com/

- Code writer + interpreter: We use GPT-5 [\[23\]](#page-11-4), GPT-5-mini [\[23\]](#page-11-4), bigcode/starcoder2-15b [\[59\]](#page-13-0), and Qwen/Qwen2.5-Coder-32B-Instruct [\[24\]](#page-11-5) as code expert models to write code. We also implemented a Python sandbox to execute the code.
- Math models: Qwen/Qwen2.5-Math-72B [\[25\]](#page-11-6), Qwen/Qwen2.5-Math-7B [\[25\]](#page-11-6)
- Generalist models: GPT-5 [\[23\]](#page-11-4), GPT-5-mini [\[23\]](#page-11-4), meta-llama/Llama-3.3-70B-Instruct [\[26\]](#page-11-7), metallama/Llama-3.1-8B-Instruct [\[26\]](#page-11-7), deepseek-ai/DeepSeek-R1 [\[57\]](#page-12-16), nvidia/Llama-3\_1-Nemotron-Ultra-253B-v1 [\[29\]](#page-11-10), microsoft/Phi-4-mini-instruct [\[58\]](#page-12-17), Qwen/Qwen3-32B [\[27\]](#page-11-8)

