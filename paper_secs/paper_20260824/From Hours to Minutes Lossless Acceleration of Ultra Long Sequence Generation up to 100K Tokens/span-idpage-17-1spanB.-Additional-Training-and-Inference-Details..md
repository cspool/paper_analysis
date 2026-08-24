# <span id="page-17-1"></span>**B. Additional Training and Inference Details.**

## **B.1. Training Details**

During training, only three linear layers are fine-tuned, while the parameters of the LLM remained fixed. The model was trained on an NVIDIA A100-SXM4-80GB GPU. The specific training parameters are outlined in Table [9.](#page-17-2)

|  | Table 9. Additional training details. Note that these hyperparameters do not require extensive tuning. |
|--|--------------------------------------------------------------------------------------------------------|
|  |                                                                                                        |

<span id="page-17-2"></span>

|                             | LLaMA3.1-8b | YaRN-LLaMA2-7b-128k | Qwen2.5-1.5b | Qwen2.5-7b | Qwen2.5-14b |
|-----------------------------|-------------|---------------------|--------------|------------|-------------|
| optimizer                   |             |                     | AdamW        |            |             |
| betas                       |             |                     | (0.9, 0.999) |            |             |
| weight decay                |             |                     | 0.1          |            |             |
| warmup steps                |             |                     | 50           |            |             |
| learning rate scheduler     |             |                     | cosine       |            |             |
| num. GPUs                   |             |                     | 4            |            |             |
| gradient accumulation steps |             |                     | 10           |            |             |
| batch size per GPU          |             | 3                   |              |            | 1           |
| num. steps                  |             | 200                 |              |            | 600         |
| learning rate               |             | 5e-3                |              |            | 1e-3        |

<span id="page-18-1"></span>Table 10. *k* stands for the maximum number of retrieved n-grams in token reutilization

|                     | k  | temp. | top-p | min-p | penalty | penalty len. |
|---------------------|----|-------|-------|-------|---------|--------------|
| LLaMA3.1-8b         |    |       | -     | 0.1   | 1.2     |              |
| YaRN-LLaMA2-7b-128k |    |       | 0.9   | -     | 1.15    |              |
| Qwen2.5-1.5b        | 20 | 1.0   | 0.9   | -     | 1.15    | 1024         |
| Qwen2.5-7b          |    |       | -     | 0.05  | 1.15    |              |
| Qwen2.5-14b         |    |       | -     | 0.05  | 1.13    |              |

## **B.2. Inference Details**

For inference, we used 4-grams to maintain consistency with multi-token generation. The specific inference parameters are presented in Table [10.](#page-18-1)

<span id="page-18-2"></span>For the tree attention mechanism, we selected a simple ternary full tree configuration, as depicted in Appendix [B.2.](#page-18-2)

![](_page_18_Figure_6.jpeg)

