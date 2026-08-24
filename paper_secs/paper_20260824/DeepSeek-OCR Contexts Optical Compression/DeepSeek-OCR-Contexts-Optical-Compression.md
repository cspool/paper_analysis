# **DeepSeek-OCR: Contexts Optical Compression**

Haoran Wei, Yaofeng Sun, Yukun Li

DeepSeek-AI

## **Abstract**

We present DeepSeek-OCR as an initial investigation into the feasibility of compressing long contexts via optical 2D mapping. DeepSeek-OCR consists of two components: DeepEncoder and DeepSeek3B-MoE-A570M as the decoder. Specifically, DeepEncoder serves as the core engine, designed to maintain low activations under high-resolution input while achieving high compression ratios to ensure an optimal and manageable number of vision tokens. Experiments show that when the number of text tokens is within 10 times that of vision tokens (i.e., a compression ratio < 10×), the model can achieve decoding (OCR) precision of 97%. Even at a compression ratio of 20×, the OCR accuracy still remains at about 60%. This shows considerable promise for research areas such as historical long-context compression and memory forgetting mechanisms in LLMs. Beyond this, DeepSeek-OCR also demonstrates high practical value. On OmniDocBench, it surpasses GOT-OCR2.0 (256 tokens/page) using only 100 vision tokens, and outperforms MinerU2.0 (6000+ tokens per page on average) while utilizing fewer than 800 vision tokens. In production, DeepSeek-OCR can generate training data for LLMs/VLMs at a scale of 200k+ pages per day (a single A100-40G). Codes and model weights are publicly accessible at http://github.com/deepseek-ai/DeepSeek-OCR.

<span id="page-0-0"></span>> **[图片提取文字 (无描述)]:**
> DeepSeek-OCR (Gundam-M 200dpi) 64 vis toks(left) 100 vis toks(left) 64 vis toks(right) 100 vis toks(right) dots.ocr (200dpi) DeepSeek-OCR (Gundam) 96.5% DeepSeek-OCR (Large) 100% 97.3% 96.8% 96.8% MinerU2.0 DeepSeek-OCR (6 e) 19.7 93.89 20x 0.2 91.5% 89.8% dots.ocr 90% Distance) o w Qwen2.5-VL-72B 85.8% High Accuracy 83.8% DeepSeek-OCR (Small) InternVL3-78B ED < 0.25 (1 better) 80% 76.3% OCRFlux-3B 15x (Edit 70% GOT-OCR2.0 Qwen2.5-VL-7B Compression (×) DeepSeek-OCR (Tiny) Precision (%) Overall Performance OLMOCR 44.00 InternVL2-76B 40% Vison Tokens > 1500 Vision Tokens < 1000 .... Average per image (← More) Average per image (→ Fewer) 30% **Encoder Series** 20% DeepEncoder Series OwenEncoder Series SmolDocling 10% InternVLEncoder Series Other Encoders 600-700 800-900 900-1000 1000-1100 700-800 200 Text Tokens in Per Page (Ground-truth) Average Vision Tokens per Image (a) Compression on Fox benchmark (b) Performance on Omnidocbench
![](_page_0_Figure_7.jpeg)

<span id="page-0-1"></span>Figure 1 | Figure (a) shows the compression ratio (number of text tokens in ground truth/number of vision tokens model used) testing on Fox [21] benchmark; Figure (b) shows performance comparisons on OmniDocBench [27]. DeepSeek-OCR can achieve state-of-the-art performance among end-to-end models enjoying the fewest vision tokens.

