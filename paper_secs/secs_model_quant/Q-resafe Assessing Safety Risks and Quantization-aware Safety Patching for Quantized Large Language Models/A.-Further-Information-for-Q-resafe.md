# A. Further Information for **Q-resafe**

In this section, we provide additional details on the training in Detailed Setup [A.1](#page-13-1) and evaluation in Details of Datasets and Corresponding Evaluations [B](#page-13-0) used in our quantization experiments. By evaluating models across different quantization settings and decoding strategies, we provide Detailed Results and Analysis in [B.](#page-15-1)

#### <span id="page-13-1"></span>A.1. Detailed setup

Our experiments were conducted on 4 NVIDIA A100 40GB GPUs, leveraging PyTorch and Hugging Face Transformers as the primary frameworks. The original model weights for Llama-2-7B-Chat and Gemma-7B-Instruct were obtained from the Hugging Face Hub.

For finetuning, we applied the following hyper-parameters:

• LoRA r: 128

• LoRA α: 256

• DPO β: 0.01

• Learning rate: 5e-6

These hyperparameters were chosen to achieve an optimal balance between training efficiency and model performance in our quantization experiments. The fine-tuning process was guided by instruction tuning, where two GPT-based APIs were used to simulate the roles of a user and an assistant for generating diverse and high-quality training pairs.

