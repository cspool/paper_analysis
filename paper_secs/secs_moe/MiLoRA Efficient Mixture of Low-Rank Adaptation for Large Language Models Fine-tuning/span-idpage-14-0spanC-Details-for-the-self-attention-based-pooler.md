# <span id="page-14-0"></span>C Details for the self-attention based pooler

Our LoRA routers must pool the input prompts of variable lengths to a fixed length. For the pooling operation, the previous literature often chooses average pooling or max pooling [\(Kim,](#page-10-22) [2014;](#page-10-22) [Zhu](#page-13-13) [et al.,](#page-13-13) [2021c;](#page-13-13) [Zhu,](#page-12-19) [2021a\)](#page-12-19), which are pointed out by the literature [\(Zhu,](#page-12-18) [2021b\)](#page-12-18) that they are prone to weaken important words when the input sequence is long, thus dropping useful information during pooling. Thus, in this work, we utilize the self-attention mechanism in our pooling module Pooler(). Self-Attention assigns each token in the input instruction a weight to indicate the importance of the token. A few crucial tokens to the task will be emphasized, while the less important tokens are ignored. Formally, we initialize a learnable weight matrix Wsa ∈ R d×1 , then the self-attention based pooler's calculation processes are:

$$\mathbf{U} = \mathbf{h}W_{sa},$$
 $\mathbf{A} = \text{Softmax}(\mathbf{U}),$ 
 $\mathbf{p} = \mathbf{A}^{\mathsf{T}}\mathbf{h},$  (8)

where p ∈ R np×d is the input tesor, Softmax is the softmax function along the first dimension, and ⊺ denotes matrix transpose. In the above equations,

each column of Wsa is a trainable query vector designated to determine the self-attention weights via dot products between this query and each token. Then, the weights are normalized across the sequence dimension via the softmax normalization function. Corresponding to different soft tokens, different query vectors in Wsa can aggregate the input instructions in different aspects, thus providing a high-quality summarization of the instruction's semantic information.

### <span id="page-14-1"></span>D Appendix for Experimental settings

Here, we provide more details for experimental settings.

Hyper-parameters for the baseline PEFT methods For P-tuning V2, the number of prompt tokens at each layer is set to 16, and the soft prompts are initialized with dimension 640, and then is projected to dimension 4096. For IAPT, the prompt length is 4, and the bottleneck dimension for the prompt generator is 320.

For the Parallel-Adapter and Learned-Adapter, the bottleneck dimension is set to 160. Adapters are connected to both the self-attention and FFN sub-layer.

We adjust the sparsity for SSP so that the number of tunable parameters is comparable with MiLoRA and the other baselines. For BitFit, the bias vectors are initialized with dimension 64, and then a learnable projection layer projects it to the same dimension with the LlaMA-2 backbone. For (IA)<sup>3</sup> , the activation adjusting vectors are added the Query,

Key, and Up activations. The adjusting vectors are initialized with dimension 128, and then a learnable projection layer projects it to the same dimension with the LlaMA-2 backbone.

For LoRA, the rank size r at each LoRA module is set to 32. For AdaLoRA, the initial rank at each module is set to 64, and half of the rank budget is pruned during fine-tuning. For MOELoRA, the rank size r at each LoRA module is set to 32, and the LoRA modules is reformulated as 32 singlerank LoRAs. Then each 4 forms an expert. Thus, a LoRA module consists of 8 experts, and the router is top-4 router, activating 4 of the expert for predicting the next token. DoRA also sets the rank size r to 32.

Training settings for PEFT methods We use the HugginFace Transformers [\(Wolf et al.,](#page-11-15) [2020b\)](#page-11-15), PEFT [\(Mangrulkar et al.,](#page-10-23) [2022\)](#page-10-23), or the original code repositories for implementing all the methods, and for training and making predictions. For fine-tuning LlaMA-2 7B model, the maximum sequence length is set to 768. The maximum training epoch is set to 10. The batch size is set between 16 for task with less than 10k training set, and 128 otherwise. We use AdamW as the optimizer with a linear learning rate decay schedule and 6% of the training steps for warm-up. The learning rate is set to 1e-4. For MiLoRA, the load balance loss coefficient λlb is set to 1e-2. For the bi-level optimization of learnable activations, the validation set is the same with the dev set. The hyper-parameters for calculating the gradients of the architectural parameters are the same with the normal training procedure, except that the learning rate is 1e-6. The other hyper-parameters are kept the same with [\(Wolf et al.,](#page-11-15) [2020b\)](#page-11-15). In every 200 steps, the model is evaluated on the dev set to calculate dev set perplexity. Patience is set to 10, that is, if the model does not achieve a lower dev set perplexity for 10 evaluation runs, the training stops early. The best checkpoint on the dev set is used to run predictions on the test set.

