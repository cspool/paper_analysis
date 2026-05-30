# B Model and Computation

LLM Model The LLM foundation model is primarily based on two families: the RWKV-5 series[1](#page-13-0) and the RWKV-6 series[2](#page-13-1) . Both the RWKV-5 and RWKV-6 series consist of models with 1.6 billion, 3 billion, and 7 billion parameters respectively. In this research, the RWKV-5 series is mainly applied in the VisualRWKV-Base, and the RWKV-6 series acts as the LLM backbone for VisualRWKV.

Model Size The vision encoders utilized in this paper are based on the CLIP-L model, which features 0.3 billion parameters. In contrast, the RWKV models vary in size: the RWKV 7B has 7.6 billion parameters, the RWKV 1.6B has 1.6 billion parameters, and the RWKV 3B has 3.1 billion parameters. Consequently, the VisualRWKV variants have different total parameter counts: the VisualRWKV 1.6B encompasses 1.9 billion parameters, the VisualRWKV 3B includes 3.4 billion parameters, and the VisualRWKV 7B comprises 7.9 billion parameters.

Computing Infrastructure A range of computational resources were employed in the study. The standard training and benchmark evaluation were conducted using 8 NVIDIA A100-80GB GPUs. The VisualRWKV 7B model is trained with 6 A100 GPUs due to insufficient memory capacity with 8 GPUs. For the efficiency analysis, a GPU with L20-48GB of memory was employed.

Computing Budget Training an epoch of VisualRWKV 1.6B with 8 A100 GPUs takes 6.7 hours, equivalent to 53.6 GPU hours; Training an epoch of VisualRWKV 3B with 8 A100 GPUs takes 11.3 hours, equivalent to 90.4 GPU hours; Training an epoch of VisualRWKV 7B with 6 A100 GPUs takes 26.5 hours, equivalent to 159 GPU hours.

Packages Version The main experimental environment for this study is the NVIDIA PyTorch NGC Container (23.07-py3) with lightning1.9.5 and deepspeed0.12.6. For updates, please refer to our codebase (currently anonymized, will be released later).

