# <span id="page-6-0"></span>3 EXPERIMENTS

## 3.1 SETTINGS

Models. We evaluate KiVi and *LogQuant* by 3 popular model families: Llama3/Llama3.1 [\(Dubey](#page-9-3) [et al., 2024\)](#page-9-3), Qwen1.5/Qwen2 [\(Bai et al., 2023;](#page-9-10) [Yang et al., 2024\)](#page-11-2), and Microsoft Phi3 [\(Abdin et al.,](#page-9-7) [2024\)](#page-9-7). Qwen1.5 and Phi3 are based on Multi-Head Attention, whereas Llama3/3.1 and Qwen2 utilize Group-Query Attention. The quantization group size G is set to the Hugging Face default value of 64, and the quantized precision is set to INT2. For KiVi, the maximum length of reserved original-precision tokens R is set to [128, 192, 256]. For LogQuant, the window length W is limited to ⌊ R 3 ⌋ as it will reserve a maximum of 3W original precision tokens to ensure that the total number of reserved original-precision tokens does not exceed that of KiVi.

Datasets. We selected GSM8K(Grade School Math, [\(Cobbe et al., 2021\)](#page-9-4)) and LongBench [\(Bai](#page-9-11) [et al., 2024\)](#page-9-11) due to their widespread use in evaluating KV cache quantization, ensuring our results are comparable to those in the literature. For GSM8K, we test with a 5-shot from the training set for better accuracy and keep the length of the input token between 600 and 1700, the evaluation is based on the exact value of the final answer. For LongBench, we test all 21 datasets among 6 types of tasks and use the LongBench's original pipeline for evaluation. The test dataset details are present in Table [B5.](#page-15-0)

