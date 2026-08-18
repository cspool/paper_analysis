# <span id="page-7-1"></span>*A. Experimental Methodology*

Models and Datasets. We select eight representative models from Llama and Mistral families with varying sizes. Specifically, we evaluate Llama-1-(7B, 13B) [\[84\]](#page-15-0); Llama-2- (7B, 13B) [\[67\]](#page-14-13); Llama-3.1-8B [\[64\]](#page-14-14); Llama-3.2-3B [\[66\]](#page-14-6); and Mistral-7B-(v0.1, v0.3) [\[38\]](#page-13-0). We obtain the pre-trained models from their HuggingFace repository, and implement the proposed W4A8KV4P8 quantization algorithm in PyTorch. We apply pre-RoPE key-cache quantization to Llama-1 and Llama-2 given their short sequence length, while Llama-3 and Mistral adopt post-RoPE key-cache quantization. For accuracy evaluation of quantized models, we measure the perplexity[3](#page-0-0) on Wikitext-2 [\[63\]](#page-14-25) and C4 [\[14\]](#page-13-20) datasets. Additionally, we evaluate the accuracy on three difficult logical and mathematical reasoning tasks, including MMLU [\[26\]](#page-13-25), ARC-Challenge [\[10\]](#page-13-26), and GSM8K [\[74\]](#page-14-27). For these reasoning tasks, we use the

<sup>3</sup>For perplexity comparison with other baseline works, we use a context length of 2K to align with their reported results. Additionally, Llama-1 supports a maximum context length of 2K; exceeding this limit results in infinite perplexity across all baselines.

TABLE IV. Wikitext-2 and C4 perplexity (↓) under different quantization methods and precisions. For both KV-cache-only and weightactivation quantization, we highlight the best perplexity results in bold.

<span id="page-8-0"></span>

|          |                |           | Llama-1 |      |      | Llama-2 | Llama-3 |        | Mistral |         |           |
|----------|----------------|-----------|---------|------|------|---------|---------|--------|---------|---------|-----------|
| Dataset  | Method†        | Precision | 7B      | 13B  | 7B   | 13B     | 3.1-8B  | 3.2-3B | 7B-v0.1 | 7B-v0.3 | Mean ∆ppl |
|          | Baseline       | FP16      | 5.68    | 5.09 | 5.47 | 4.88    | 6.24    | 7.81   | 5.25    | 5.32    | 0         |
|          | Oaken          | KV4       | 5.75    | 5.15 | 5.53 | 4.93    | 6.37    | 8.01   | 5.28    | 5.35    | 0.08      |
|          | 3<br>P<br>-LLM | KV4       | 5.72    | 5.12 | 5.51 | 4.92    | 6.35    | 7.96   | 5.29    | 5.35    | 0.06      |
| Wikitext | QuaRot         | W4A8KV4   | 5.92    | 5.28 | 5.70 | 5.07    | 6.80    | 8.48   | 5.41    | 5.49    | 0.30      |
|          | QoQ            | W4A8KV4   | 5.89    | 5.24 | 5.69 | 5.07    | 6.79    | 8.56   | 5.42    | 5.50    | 0.30      |
|          | 3<br>P<br>-LLM | W4A8KV4P8 | 5.81    | 5.22 | 5.65 | 5.01    | 6.75    | 8.39   | 5.41    | 5.48    | 0.25      |
|          | Baseline       | FP16      | 7.08    | 6.61 | 6.97 | 6.47    | 8.96    | 10.43  | 7.74    | 7.83    | 0         |
|          | Oaken          | KV4       | 7.15    | 6.66 | 7.04 | 6.52    | 9.10    | 10.69  | 7.79    | 7.87    | 0.09      |
|          | 3<br>P<br>-LLM | KV4       | 7.11    | 6.63 | 7.01 | 6.50    | 9.09    | 10.62  | 7.78    | 7.87    | 0.07      |
| C4       | QuaRot         | W4A8KV4   | 7.31    | 6.77 | 7.26 | 6.68    | 9.95    | 11.73  | 8.05    | 8.16    | 0.48      |
|          | QoQ            | W4A8KV4   | 7.28    | 6.75 | 7.22 | 6.65    | 9.72    | 11.57  | 7.94    | 8.04    | 0.38      |
|          | 3<br>P<br>-LLM | W4A8KV4P8 | 7.23    | 6.72 | 7.14 | 6.59    | 9.64    | 11.31  | 7.94    | 8.03    | 0.31      |

<sup>†</sup> For weight-activation quantization, QuaRot and QoQ maintain query and attention-scores in FP16. Whereas P<sup>3</sup> -LLM quantizes attention-scores to FP8-S0E4M4 across all models, and quantizes query to FP8-E4M3 for Llama-3 and Mistral.

instruction-tuned variants of Llama-3.1-8B and Llama-3.2-3B as suggested in the LM-Eval framework [\[19\]](#page-13-27).

Algorithm Baselines. We compare P<sup>3</sup> -LLM with SoTA quantization algorithms, including Oaken [\[42\]](#page-14-16), QuaRot [\[2\]](#page-13-9), and QoQ [\[53\]](#page-14-20). Oaken is a 4-bit KV-cache quantization method that determines the key-cache outlier threshold offline using a calibration dataset. We compare Oaken with P<sup>3</sup> -LLM under KV-cache-only quantization for a fair comparison. QuaRot and QoQ implement aggressive W4A8KV4 quantization using the standard integer format. For all baseline algorithms, we use their official GitHub repository to reproduce the accuracy.

Hardware Implementation. We model P<sup>3</sup> -LLM's PCU at RTL-level using SystemVerilog and synthesize it using Synopsys Design Compiler under TSMC 28nm technology. We scale the area and power results from our RTL simulation to HBM-PIM's 20nm technology using DeepScaleTool [\[80\]](#page-15-13). We further scale the area overhead by taking into account the effects of DRAM process, which has 10× lower transistor density than the logic process under the same feature size [\[13\]](#page-13-6). For performance evaluation, we develop a cycle-level simulator to model the P<sup>3</sup> -LLM system with 4 NPU cores and 16 pseudo HBM channels. The NPU design is based on [\[27\]](#page-13-3), where each core contains a 128×128 systolic array, a 128 way vector processing unit, and 16MB on-chip scratchpad that is modeled with CACTI [\[5\]](#page-13-28). The PIM subsystem is simulated following the methodology of Newton [\[23\]](#page-13-29), which is extended to model the computation flow of the proposed PCU. We set the operational frequency of NPU and PCU as 1 GHz and 500 MHz, respectively, considering tCCD <sup>S</sup> of HBM2 that is 2 DRAM clock cycles.

Accelerator Baselines. To evaluate hardware performance, we compare P<sup>3</sup> -LLM against three baselines: (1) An NPU accelerator running FP16 models, without PIM support in its DRAM; (2) An NPU accelerator integrated with HBM-PIM [\[49\]](#page-14-9) running FP16 models; (3) A SoTA LLM accelerator, Ecco [\[8\]](#page-13-10), employing W4A8KV4 quantization with k-means codebooks and Huffman encoding. All baseline systems share the same NPU and DRAM configurations as P<sup>3</sup> -LLM with comparable total area. For comparison of accelerator performance, we focus on the decoding stage that typically dominates the endto-end inference latency of edge applications [\[20\]](#page-13-19), [\[52\]](#page-14-4). We use a context length of 4K that covers most edge scenarios as suggested in prior works [\[16\]](#page-13-11), [\[50\]](#page-14-3). Notably, for accelerator evaluation, we only assess Llama-2-(7B, 13B), Llama-3.1- 8B, Llama-3.2-3B, and Mistral-7B-v0.3, since the remaining models share the same size and/or architecture.

