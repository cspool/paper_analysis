# **D** Experimental settings

Computing Infrastructure We run all our experiments on NVIDIA A6000 (48GB) GPUs, using Python 3.10 and Ubuntu 20.04 on x86-64 CPUs.

**Pretrained Backbones** The main experiments use the most recent open-sourced LLM, LLaMA-2 7B and Gemma 2B, as the pretrained backbone model. When fine-tuning LLaMA-2 7B and Gemma 2B, we consider only the supervised fine-tuning setting.

Hyperparameters for CoMoE In our experiments, unless otherwise specified, we set the hyperparameters as illustrated in Table 6. In the table, the hyperparameters set by other baseline methods, LoRA, DoRA, MixLoRA, MixDoRA, OMoELoRA, are also included. Under this setting, Co-MoE introduces approximately 1.45% tunable parameters to the LLaMA-2 7B backbone.

**Descriptive Statistics about Results** We conduct experiments on all training settings using five different random seeds, and the final results represent the median accuracy within each setting.

