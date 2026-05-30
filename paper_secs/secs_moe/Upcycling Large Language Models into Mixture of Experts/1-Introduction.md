# 1 Introduction

Sparse Mixture of Experts (MoE) models [\[1\]](#page-14-0) are becoming increasingly popular [\[2,](#page-14-1) [3,](#page-14-2) [4,](#page-14-3) [5,](#page-14-4) [6,](#page-14-5) [7\]](#page-14-6) since they can help achieve better accuracy without a commensurate increase in model training compute. Most recently, state-of-the-art LLMs like Grok-1[2](#page-0-1) , DBRX[3](#page-0-2) , Phi-3.5[4](#page-0-3) , Mixtral 8x22B [\[3\]](#page-14-2), DeepSeek-V2 [\[8\]](#page-14-7) and Qwen2 [\[9\]](#page-14-8) are MoE models. However, an immense amount of compute has been spent on pre-training dense LLMs with only one MLP layer (one expert) [\[10,](#page-14-9) [11,](#page-14-10) [12,](#page-14-11) [13,](#page-14-12) [14\]](#page-14-13). These existing dense models may be able to achieve better accuracy for the same compute cost if they had access to more parameters through MoEs. Upcycling pre-trained dense language models into sparse mixture-ofexperts models (referred to as simply 'upcycling' in this work) has emerged as an efficient approach to increase model capacity without the need for training from scratch [\[15,](#page-15-0) [16,](#page-15-1) [17,](#page-15-2) [3\]](#page-14-2). By leveraging the knowledge captured in existing dense checkpoints, upcycling enables the creation of large-scale mixtureof-experts (MoE) models while reducing the computational cost and time required for training.

Most previous work on upcycling either does not provide details into how their models were upcycled [\[3\]](#page-14-2), or provides experiments only at a small scale [\[15\]](#page-15-0). We also find that the recommendations in [\[9\]](#page-14-8) lead to sub-optimal models. To improve general knowledge on upcycling methods, we therefore publish this study of upcycling methods and best practices for billion-parameter scale language models.

In this work, we conduct an extensive study of upcycling techniques and hyperparameters. Our contributions are as follows:

- 1. We recommend training recipes to consistently upcycle billion-parameter scale LLMs.
- 2. We perform a comprehensive study to find the best hyperparameters for upcycling including learning rate, batch size, and load balancing loss.

<span id="page-0-0"></span><sup>∗</sup>equal contribution. Correspondence to: {yihuih,akhattar,rprenger}@nvidia.com

<sup>1</sup>[https://github.com/NVIDIA/Megatron-LM/tree/0431153bf1b5c405057b158189c260107d8b7c3a/megatron/core/](https://github.com/NVIDIA/Megatron-LM/tree/0431153bf1b5c405057b158189c260107d8b7c3a/megatron/core/transformer/moe#upcycling) [transformer/moe#upcycling](https://github.com/NVIDIA/Megatron-LM/tree/0431153bf1b5c405057b158189c260107d8b7c3a/megatron/core/transformer/moe#upcycling)

<span id="page-0-1"></span><sup>2</sup>https://x.ai/blog/grok-os

<span id="page-0-2"></span><sup>3</sup>https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm

<span id="page-0-3"></span><sup>4</sup>https://huggingface.co/microsoft/Phi-3.5-MoE-instruct

- 3. We propose a novel "virtual group" initialization scheme to enable upcycling into fine-grained MoE architectures, along with a weight scaling approach which brings 1.5% better loss to both coarse-grained and fine-grained upcycled MoE models.
- 4. We compare softmax-then-topK expert routing with the topK-then-softmax approach.
- 5. We assess the benefits of higher granularity MoE models and using higher topK.

We demonstrate that our upcycling approach produces a better model than continued dense model training, softmax-then-topK routing improves over topK-then-softmax approach, and higher granularity MoEs can help boost model accuracy in certain training scenarios. Finally, we upcycle the Nemotron-4 15B model [\[13\]](#page-14-12) into MoE on 1T tokens and show that it improves MMLU. To make the comparison fair, we train the Nemotron-4 model for an additional 1T tokens and achieve a 65.3% MMLU score, whereas our Nemotron-4 model which was upcycled on the same 1T tokens achieves 67.6. This shows that the improvement from upcycling is not just due to the additional tokens the model is trained on, but also due to the MoE architecture.

By sharing our upcycling approach and ablation results, we aim to contribute to the growing body of knowledge on efficient and scalable language model development, enabling researchers and practitioners to build upon our work and further advance the field of large-scale MoE models. We use Megatron-LM[5](#page-1-0) [\[18\]](#page-15-3) to upcycle and train our MoE models.

