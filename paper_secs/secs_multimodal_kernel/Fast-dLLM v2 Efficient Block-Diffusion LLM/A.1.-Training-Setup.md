# A.1. Training Setup

We fine-tune pretrained Qwen2.5-Instruct models (1.5B and 7B) under our block-wise diffusion training framework. Unless otherwise specified, all experiments adopt a context length of 2048 and batch size of 256. Training is conducted on 64 NVIDIA A100 GPUs using DeepSpeed Zero-3.

**Training Data.** Our models are fine-tuned on a subset of the LLaMA-Nemotron post-training dataset, which contains high-quality instruction-following examples covering a broad range of domains. We preprocess the dataset using block-wise packing, and pad each sequence to a multiple of the block size to avoid misaligned block boundaries. Redundant padding tokens are excluded from loss computation and gradient updates.

**Hyperparameters.** The 1.5B model is trained for 6,000 steps with a learning rate of  $2 \times 10^{-5}$ , while the 7B model is trained for 2,500 steps with a learning rate of  $1 \times 10^{-5}$ . In both settings, we use AdamW as the optimizer and apply linear learning rate warmup over the first 500 steps. With a context length of 2048 and batch size of 256, each training step processes  $256 \times 2048 = 524,288$  tokens. This corresponds to a total training token count of approximately:

- 1.5B model:  $6,000 \times 524,288 \approx 3.15$  billion tokens
- 7B model:  $2,500 \times 524,288 \approx 1.31$  billion tokens

We fix the block size to 32 for all experiments. All training sequences are right-padded and packed in a block-aligned fashion to fully utilize model context, enabling consistent and efficient batch construction under hardware constraints.

#### A.2. Attention Mask Design

<span id="page-12-0"></span>![](_page_12_Figure_10.jpeg)

Figure 7 | Specialized attention mask design for diffusion language modeling. (a) During training, each input consists of a corrupted sequence  $x_t$  and corresponding targets  $x_0$ , concatenated and processed in a single forward pass. The attention mask combines intra-block bidirectional attention (Block Diagonal), cross-block causal dependency from clean tokens to noised ones (Offset Block Causal), and traditional left-to-right causality among clean tokens (Block Causal). (b) During inference, previously decoded blocks of  $x_0$  are reused via caching. Only the current noised block  $x_t$  is computed in each decoding step, which attends to cached prefixes (shaded) and updates its own block in a self-contained fashion.

To enable efficient and structured learning across both corrupted and clean views of the input, we use a custom block-aware attention scheme (Arriola et al., 2025). At each training step, we concatenate the noised sequence  $x_t$  and the clean sequence  $x_0$  into a single input of total length 2L, then apply a hybrid attention pattern defined via an attention mask  $\mathcal{M}_{\text{full}} \in \{0,1\}^{2L \times 2L}$ .

To simplify notation, we follow prior work and slightly abuse the symbol  $x^b$ , which in this context denotes the set of tokens in the b-th block (rather than the b-th token, as in earlier sections). Specifically, we aim to model the conditional probabilities  $p_{\theta}(x^b \mid x_t^b, x^{< b})$  across all blocks  $b \in [1, B]$ , where  $x_t^b$  is the noised version of block b, and  $x^{< b}$  comprises all clean tokens in previous blocks. This formulation enables us to process both noised and clean representations simultaneously by feeding their concatenated sequence into the transformer and applying a carefully constructed attention mask  $\mathcal{M}_{\text{full}}$  as shown in Figure 7(a).

The overall attention mask can be decomposed into four sub-masks:

$$\mathcal{M}_{\mathrm{full}} = \begin{bmatrix} \mathcal{M}_{BD} & \mathcal{M}_{OBC} \\ 0 & \mathcal{M}_{BC} \end{bmatrix},$$

where:

•  $\mathcal{M}_{BD}$  (Block-diagonal mask): Provides bidirectional self-attention among tokens within the same block in the noised sequence  $x_t$ , enabling within-block refinement:

$$[\mathcal{M}_{BD}]_{ij} = \begin{cases} 1 & \text{if } i,j \text{ belong to the same block} \\ 0 & \text{otherwise} \end{cases}$$

•  $\mathcal{M}_{OBC}$  (Offset block-causal mask): Allows each noised token in  $x_t$  to attend to tokens from previous blocks in the clean sequence  $x_0$ , preserving inter-block causal conditioning:

$$[\mathcal{M}_{OBC}]_{ij} = \begin{cases} 1 & \text{if } j \text{ is in a block before } i \\ 0 & \text{otherwise} \end{cases}$$

•  $\mathcal{M}_{BC}$  (Block-causal mask): Enables each token in the clean sequence  $x_0$  to attend to all previous and current block positions, facilitating autoregressive-like progression:

$$[\mathcal{M}_{BC}]_{ij} = \begin{cases} 1 & \text{if } j \text{ is in the same or an earlier block as } i \\ 0 & \text{otherwise} \end{cases}$$

The combined mask allows unified handling of masked token prediction, simultaneous conditioning on prior known context, and structural training efficiency via block-parallelism.

During inference, we adopt a simplified causal attention mechanism that reuses decoded blocks as frozen prefix context. As illustrated in Figure 7(b), previously generated blocks from  $x_0^{<b}$  are cached to avoid redundant computation, and only the current noised block  $x_t^b$  is actively refined. This block attends bidirectionally within itself, similar to  $\mathcal{M}_{BD}$  during training, while attending causally to the unmasked tokens in previous blocks. The attention computation is thus restricted to the current block and its causal prefix, enabling efficient decoding via key-value cache reuse and reduced memory footprint. This structure preserves left-to-right semantics across blocks while allowing intra-block denoising in parallel.

### A.3. Details on Training Objective

We minimize the masked-token-only cross-entropy loss:

$$\mathcal{L}_{\text{block}}(\theta) = -\mathbb{E}_{x,m} \left[ \sum_{i=1}^{L} \mathbf{1}[x_t^i = [\text{MASK}]] \log p_{\theta}(x_0^i \mid x_{< i}, x_{\text{block}(i)}) \right].$$

Notably, this objective function seems to omit the normalization coefficient  $\frac{1}{t}$  often found in standard masked modeling losses (e.g., dividing by the number of masked tokens). This is intentional and justified by our complementary masking strategy. This is because we use a complementary mask for each training sample  $x_0$ : we always sample two complementary times t and t0 and t1 and t2 with mask t3 and t4 and t5 with mask t5 and t5 and t6 are training sample t6.

$$-\left[\sum_{i=1}^{L}\mathbf{1}[x_{t}^{i} = \texttt{[MASK]}]\log p_{\theta}(x_{0}^{i} \mid x_{< i}, x_{\mathsf{block}(i)})\right] + \left[\sum_{i=1}^{L}\mathbf{1}[x_{1-t}^{i} = \texttt{[MASK]}]\log p_{\theta}(x_{0}^{i} \mid x_{< i}, x_{\mathsf{block}(i)})\right].$$

Due to the complementary mask, the total number of tokens contributing to the loss for any given sample  $x_0$  is always the full sequence length L.

### A.4. Evaluation Protocol

We evaluate all trained models on a diverse suite of downstream benchmarks covering reasoning, knowledge, and code generation. Unless otherwise specified, all evaluations are conducted using greedy decoding (argmax). We adopt zero-shot settings for all tasks, with the exception of GPQA, which is evaluated under 5-shot prompting following standard protocol.

All non-code tasks are evaluated using the LM-Eval harness, ensuring compatibility and fair performance reporting. For code tasks like HumanEval and MBPP, we employ the EvalPlus framework for reliable pass-rate calculation. Unless otherwise noted, the following setup is used during inference:

- Block size = 32
- Sub-block size = 8
- Parallel decoding disabled (threshold = 1)

This configuration ensures consistency between training and inference setups, facilitating effective evaluation of the block-wise diffusion capability in Fast-dLLM v2.

