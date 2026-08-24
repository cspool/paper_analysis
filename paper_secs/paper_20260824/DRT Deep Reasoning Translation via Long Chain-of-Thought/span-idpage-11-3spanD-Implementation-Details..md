# <span id="page-11-3"></span>D Implementation Details.

Automatic Evaluation. To calculate CometKiwi and CometScore, we leverage the official codes[7](#page-11-4) and the official models[8](#page-11-5) . To calculate the BLEU score, we use the *sacrebleu* toolkit[9](#page-11-6) to calculate the corpus-level BLEU.

Training Details. Llama-Factory [\(Zheng et al.,](#page-9-16) [2024\)](#page-9-16) is used to instruct-tune LLMs. All LLMs are tuned on 8×NVIDIA A100 GPUs (40G) with 1e-5 learning rate and 8 (8×1) batch size. We use the DeepSpeed ZeRO-3 optimization [\(Rasley et al.,](#page-9-17) [2020\)](#page-9-17). Following [Qin et al.](#page-9-1) [\(2024\)](#page-9-1), we set the number of training epochs to 3, and the training process costs 70 GPU hours and 124 GPU hours for 7B and 14B models, respectively.

Inference Details. When evaluating model performance on the test set, we use vLLM toolkit [\(Kwon](#page-9-18) [et al.,](#page-9-18) [2023\)](#page-9-18) to accelerate the model generation. We

<span id="page-11-5"></span><span id="page-11-4"></span><sup>7</sup><https://github.com/Unbabel/COMET>

<sup>8</sup>[https://huggingface.co/Unbabel/](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) [wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) and [https://huggingface.](https://huggingface.co/Unbabel/wmt22-comet-da) [co/Unbabel/wmt22-comet-da](https://huggingface.co/Unbabel/wmt22-comet-da)

<span id="page-11-6"></span><sup>9</sup><https://github.com/mjpost/sacrebleu>

<span id="page-12-2"></span>

| Model                                    | reference-free |              |           | reference-based |              |              |  |
|------------------------------------------|----------------|--------------|-----------|-----------------|--------------|--------------|--|
| 112001                                   | GEA            | GRF          | CometKiwi | GRB             | BLEU         | CometScore   |  |
| Commercial LLMs                          |                |              |           |                 |              |              |  |
| GPT-40                                   | 71.88          | 85.57        | 73.01     | 82.78           | 34.51        | 79.41        |  |
| o1-preview                               | 78.01          | <u>87.11</u> | 73.70     | 83.86           | 30.65        | 80.12        |  |
|                                          |                | DRT          |           |                 |              |              |  |
| DRT-8B (Backbone: Llama-3.1-8B-Instruct) | 69.65          | 84.49        | 70.85     | 80.80           | 32.67        | 78.81        |  |
| DRT-7B (Backbone: Qwen2.5-7B-Instruct)   | 75.05          | 85.57        | 71.78     | 82.38           | <u>35.54</u> | <u>80.19</u> |  |
| DRT-14B (Backbone: Qwen2.5-14B-Instruct) | <u>77.41</u>   | 87.19        | 72.11     | <u>83.20</u>    | 36.46        | 80.64        |  |

Table 6: Experimental results of comparing DRT with commercial LLMs. The **bold** and the <u>underline</u> denote the best and second-best performances, respectively.

use the sampling decoding strategy with 0.1 temperature, and set the repetition penalty to 1.05. For DeepSeek-R1 series (DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-14B and DeepSeek-R1-Distill-Qwen-32B), we follow the instruction <sup>10</sup> to enforce them to avoid blank thinking. All experimental results listed in this paper are the average of 3 runs.

