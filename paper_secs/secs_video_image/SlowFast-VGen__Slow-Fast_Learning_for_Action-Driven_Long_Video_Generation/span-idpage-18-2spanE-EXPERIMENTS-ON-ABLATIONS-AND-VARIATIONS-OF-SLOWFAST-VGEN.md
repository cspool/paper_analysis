# <span id="page-18-2"></span>E EXPERIMENTS ON ABLATIONS AND VARIATIONS OF SLOWFAST-VGEN

We introduce several variations of SLOWFAST-VGEN, including:

<span id="page-19-0"></span>

|                               | SCuts ↓ | SRC ↑ |
|-------------------------------|---------|-------|
| Our (w original TEMP-LORA)    | 0.55    | 92.24 |
| Ours (wo Local Learning Rule) | 0.36    | 90.27 |
| Ours (wo Chunk Input)         | 1.24    | 90.01 |
| Ours (wo/ Temp-LoRA)          | 1.88    | 89.04 |
| Ours SLOWFAST-VGEN            | 0.37    | 93.71 |

Table 4: Scene Cuts and SRC Scores. Comparison of scene cuts and SRC scores for our method with and without Temp-LoRA.

- Ours wo Chunk Input that only conditions on single-frame images instead of previous chunk
- Ours wo Local Learning Rule that samples over the whole generated sequence for training TEMP-LORA, instead of using local inputs and outputs to train.
- Ours w original TEMP-LORA that uses the original TEMP-LORA structure that were designed for long text generation.

We show the results below. From the table, we can see that SLOWFAST-VGEN trained over sampled full sequence also shows good performances. However, our observation indicates that this method tends to over-smooth the generated sequences, leading to blurry videos for later frames.

