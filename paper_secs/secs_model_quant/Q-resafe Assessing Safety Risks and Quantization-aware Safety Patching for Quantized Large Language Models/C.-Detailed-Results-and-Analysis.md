# C. Detailed Results and Analysis

#### <span id="page-15-0"></span>C.1. Ablation on the patched quantization methods

We further conducted experiments on three prevalent algorithms: LLM.int8(), NF4, and FP4 [\(Dettmers et al.,](#page-9-22) [2022;](#page-9-22) [Wolf,](#page-11-21) [2020;](#page-11-21) [Dettmers et al.,](#page-9-7) [2024\)](#page-9-7), commonly used in the bitsandbytes library [\(Foundation,](#page-9-23) [2025\)](#page-9-23), to show the proposed methods can patch effectively diverse quantization methods. The results is shwon in Tab. [10,](#page-16-0) indicating a substantial degradation in safety after quantization, with ASR values reaching as high as 35.2% for FP4. However, when Q-resafe method is applied, the safety of the models is significantly restored, with ASR values dropping to as low as 5.2% for LLM.int8().

While these quantization methods improve computational efficiency, they also introduce safety vulnerabilities, making models more susceptible to adversarial attacks. This is reflected in the high ASR values observed before applying the safety patch.

These results underscore that quantization alone is insufficient for maintaining safety in low-bit models. The degradation in safety performance suggests that lower-bit models are more susceptible to adversarial attacks. However, Q-resafe successfully mitigates these vulnerabilities, ensuring that quantized models retain safety properties comparable to their full-precision counterparts. Thus, Q-resafe is not only method-agnostic but also highly effective in restoring model safety while preserving the computational benefits of quantization.

<span id="page-16-0"></span>Table 10. ASR results before and after applying the Q-resafe safety patch on popular quantization methods.

|            | w.o. Safety Patch | w. Safety Patch |
|------------|-------------------|-----------------|
| LLM.int8() | 19.2              | 5.2             |
| NF4        | 23.9              | 5.5             |
| FP4        | 35.2              | 6.0             |

#### C.2. Why fine-tuning impacts safety

To systematically assess the impact of fine-tuning on both safety and utility, we consider three different risk levels: (1) High Risk (Risk-III): We fine-tune aligned LLMs on 10, 50, and 100 harmful examples for 5 epochs. After fine-tuning, we measure ASR (%) to assess safety risks. To evaluate utility, we report the MT-Bench score and AlpacaEval after an additional 5 epochs of fine-tuning with 100 harmful examples. (2) Moderate Risk (Risk-II): We fine-tune pre-quantization LLMs on 10 identity-shifting examples and assess their post-fine-tuning safety by measuring ASR (%) for the quantized models. Utility is evaluated based on MT-Bench and AlpacaEval, measured after 10 epochs of fine-tuning. (3) Low Risk (Risk-I): We fine-tune aligned LLMs on a benign dataset (UltraChat) for 1 epoch and assess the inherent safety degradation using ASRV anilla(%). To evaluate utility under adversarial conditions, we further fine-tune the models on 100 harmful examples and report their MT-Bench score and AlpacaEval.

The results in Tables [11,](#page-16-1) [12,](#page-17-0) and [13](#page-17-1) demonstrate that Q-resafe effectively maintains a low safety risk score while preserving strong utility, even across varying fine-tuning conditions. Moreover, our findings suggest that standard alignment techniques alone are insufficient to counteract the vulnerabilities introduced by fine-tuning. Regardless of the strength of the base aligned model, fine-tuning attacks can still compromise safety and degrade its defenses. This underscores the necessity of our method in maintaining alignment robustness even under adversarial training conditions.

<span id="page-16-1"></span>Table 11. Safety and utility comparison of fine-tuned LLMs on Risk-III examples: Few-shot (10, 50, 100) and 5-Epoch training.

| Llama    | Bit-  | Size |         |         | Safety(↓) |          |          | Utility(↑) |
|----------|-------|------|---------|---------|-----------|----------|----------|------------|
| Method   | width | (GB) | Initial | 10-shot | 50-shot   | 100-shot | MT-Bench | AlpacaEval |
| Baseline | FP16  | 12.6 | 0.3     | 50.0    | 80.0      | 80.3     | 6.65     | 71.37      |
| AQLM     | 4-bit | 2.8  | -       | 77.4    | 80.5      | 81.9     | 6.50     | 66.42      |
| LLM-QAT  | 4-bit | 3.5  | -       | 71.2    | 92.6      | 93.8     | 6.52     | 66.54      |
| QLoRA    | 4-bit | 2.8  | -       | 85.3    | 94.2      | 95.7     | 6.42     | 63.92      |
| Q-resafe | 4-bit | 3.5  | -       | 13.5    | 13.9      | 14.1     | 6.59     | 68.51      |
| AQLM     | 8-bit | 6.0  | -       | 75.3    | 78.4      | 80.0     | 6.54     | 68.85      |
| LLM-QAT  | 8-bit | 6.5  | -       | 65.4    | 88.3      | 87.2     | 6.58     | 69.47      |
| QLoRA    | 8-bit | 6.0  | -       | 83.2    | 90.4      | 92.1     | 6.40     | 64.05      |
| Q-resafe | 8-bit | 6.5  | -       | 12.1    | 12.6      | 13.2     | 6.61     | 70.93      |
| Gemma    | Bit-  | Size |         |         | Safety(↓) |          |          | Utility(↑) |
| Method   | width | (GB) | Initial | 10-shot | 50-shot   | 100-shot | MT-Bench | AlpacaEval |
| Baseline | FP16  | 17.1 | 9.2     | 42.3    | 68.9      | 70.0     | 6.25     | 66.53      |
| AQLM     | 4-bit | 2.8  | -       | 55.4    | 65.7      | 66.0     | 6.10     | 61.75      |
| LLM-QAT  | 4-bit | 3.5  | -       | 52.9    | 74.2      | 75.9     | 6.19     | 62.85      |
| QLoRA    | 4-bit | 2.8  | -       | 61.3    | 70.7      | 70.9     | 6.05     | 59.13      |
| Q-resafe | 4-bit | 3.5  | -       | 10.4    | 10.7      | 11.0     | 6.21     | 63.77      |
| AQLM     | 8-bit | 6.0  | -       | 53.8    | 61.6      | 63.4     | 6.20     | 63.59      |
| LLM-QAT  | 8-bit | 6.5  | -       | 50.1    | 73.5      | 74.3     | 6.24     | 64.12      |
| QLoRA    | 8-bit | 6.0  | -       | 58.9    | 68.5      | 70.6     | 6.11     | 62.50      |
| Q-resafe | 8-bit | 6.5  | -       | 9.8     | 10.3      | 10.7     | 6.24     | 66.10      |

<span id="page-17-0"></span>Table 12. Safety and utility comparison of fine-tuned LLMs on Risk-II examples: 10-Shot learning with (3, 5, 10)-epoch training.

| Llama             | Bit-           | Size       |           |              | Safety(↓)    |              |              | Utility(↑)     |
|-------------------|----------------|------------|-----------|--------------|--------------|--------------|--------------|----------------|
| Method            | width          | (GB)       | Initial   | 3-epochs     | 5-epochs     | 10-epochs    | MT-Bench     | AlpacaEval     |
| Baseline          | FP16           | 12.6       | 0.3       | 54.2         | 72.1         | 68.2         | 6.65         | 71.37          |
| AQLM              | 4-bit          | 2.8        | -         | 60.3         | 74.2         | 75.5         | 6.60         | 67.50          |
| LLM-QAT           | 4-bit          | 3.5        | -         | 70.5         | 85.3         | 82.9         | 6.61         | 67.26          |
| QLoRA             | 4-bit          | 2.8        | -         | 78.4         | 84.9         | 83.4         | 6.20         | 67.60          |
| Q-resafe          | 4-bit          | 3.5        | -         | 12.2         | 13.4         | 13.6         | 6.63         | 67.88          |
| AQLM              | 8-bit          | 6.0        | -         | 58.0         | 70.9         | 73.3         | 6.57         | 69.20          |
| LLM-QAT           | 8-bit          | 6.5        | -         | 68.2         | 77.4         | 76.1         | 6.64         | 69.51          |
| QLoRA             | 8-bit          | 6.0        | -         | 75.2         | 77.8         | 76.7         | 6.37         | 69.50          |
| Q-resafe          | 8-bit          | 6.5        | -         | 10.5         | 11.8         | 11.2         | 6.65         | 70.06          |
|                   |                |            | Safety(↓) |              | Utility(↑)   |              |              |                |
| Gemma             | Bit-           | Size       |           |              |              |              |              |                |
| Method            | width          | (GB)       | Initial   | 3-epochs     | 5-epochs     | 10-epochs    | MT-Bench     | AlpacaEval     |
| Baseline          | FP16           | 17.1       | 9.2       | 38.5         | 57.9         | 59.1         | 6.25         | 66.53          |
|                   |                |            |           |              |              |              |              |                |
| AQLM              | 4-bit          | 2.8        | -         | 50.1         | 68.5         | 69.9         | 6.30         | 64.41          |
| LLM-QAT           | 4-bit          | 3.5        | -         | 45.3         | 66.5         | 68.4         | 6.19         | 63.01          |
| QLoRA<br>Q-resafe | 4-bit<br>4-bit | 2.8<br>3.5 | -<br>-    | 61.4<br>14.1 | 70.9<br>14.9 | 68.6<br>14.7 | 6.13<br>6.19 | 64.10<br>63.85 |
|                   |                |            |           |              |              |              |              |                |
| AQLM<br>LLM-QAT   | 8-bit<br>8-bit | 6.0<br>6.5 | -<br>-    | 45.8<br>41.8 | 62.0<br>62.9 | 60.4<br>63.5 | 6.12<br>6.22 | 63.40<br>64.94 |
| QLoRA             | 8-bit          | 6.0        | -         | 59.3         | 68.1         | 64.0         | 6.20         | 64.91          |

<span id="page-17-1"></span>Table 13. Safety and utility comparison of fine-tuned LLMs on Risk-I examples (UltraChat) after 1 epoch training.

| Llama    | Bit-  | Size | Safety (↓) |                   | Utility (↑) |            |
|----------|-------|------|------------|-------------------|-------------|------------|
| Method   | width | (GB) | Initial    | After fine-tuning | MT-Bench    | AlpacaEval |
| Baseline | FP16  | 12.6 | 0.3        | -                 | 6.65        | 71.37      |
| AQLM     | 4-bit | 2.8  | -          | 18.5              | 6.40        | 67.20      |
| LLM-QAT  | 4-bit | 3.5  | -          | 16.9              | 6.71        | 66.50      |
| QLoRA    | 4-bit | 2.8  | -          | 42.5              | 6.44        | 63.90      |
| Q-resafe | 4-bit | 3.5  | -          | 1.8               | 7.14        | 69.70      |
| AQLM     | 8-bit | 6.0  | -          | 17.1              | 6.45        | 69.10      |
| LLM-QAT  | 8-bit | 6.5  | -          | 15.1              | 6.64        | 67.80      |
| QLoRA    | 8-bit | 6.0  | -          | 41.73             | 6.37        | 65.20      |
| Q-resafe | 8-bit | 6.5  | -          | 1.6               | 7.29        | 70.84      |
| Gemma    | Bit-  | Size | Safety (↓) |                   | Utility (↑) |            |
| Method   | width | (GB) | Initial    | After fine-tuning | MT-Bench    | AlpacaEval |

| Gemma    | Bit-  | Size | Safety (↓) |                   | Utility (↑) |            |
|----------|-------|------|------------|-------------------|-------------|------------|
| Method   | width | (GB) | Initial    | After fine-tuning | MT-Bench    | AlpacaEval |
| Baseline | FP16  | 17.1 | 9.2        | -                 | 6.25        | 66.53      |
| AQLM     | 4-bit | 2.8  | -          | 25.3              | 6.12        | 62.70      |
| LLM-QAT  | 4-bit | 3.5  | -          | 20.7              | 6.28        | 63.40      |
| QLoRA    | 4-bit | 2.8  | -          | 39.1              | 6.15        | 62.40      |
| Q-resafe | 4-bit | 3.5  | -          | 10.1              | 6.75        | 66.32      |
| AQLM     | 8-bit | 6.0  | -          | 23.8              | 6.23        | 63.20      |
| LLM-QAT  | 8-bit | 6.5  | -          | 18.4              | 6.39        | 64.70      |
| QLoRA    | 8-bit | 6.0  | -          | 37.1              | 6.27        | 62.40      |
| Q-resafe | 8-bit | 6.5  | -          | 9.8               | 6.82        | 66.40      |

#### C.3. Why decoding strategies impacts safety

Decoding strategies play a crucial role in shaping a model's response behavior, influencing not only fluency and diversity but also safety and robustness. While quantization methods like AWQ enhance computational efficiency, they do not inherently preserve safety constraints, leaving models vulnerable to adversarial inputs. Figure [3](#page-18-0) show evaluation demonstrates that modifying decoding parameters can significantly impact a model's susceptibility to harmful prompts. This highlights the need for decoding-aware safety mechanisms to ensure safe and reliable model outputs.

![](_page_18_Figure_3.jpeg)

<span id="page-18-0"></span>Figure 3. Safety evaluation of the Llama2-7b-chat model under different quantization methods (INT4) and sampling strategies across 11 safety categories aligned with OpenAI's policy [\(Ope,](#page-8-11) [2023\)](#page-8-11).