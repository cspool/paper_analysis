# **A.1.2 Worked Examples**

<span id="page-19-0"></span>**Memento refinement example** — NBA playoff probability trace, Block 2 (876 tokens → memento)

**Block excerpt:** *". . . defines f(n,a,b) as the probability that after n games, Team A has a wins and Team B has b wins. Starting point: f(0,0,0)=1. Need f(6,3,3). For Game n: if n is odd, home=A, P(A wins)=0.6; if even, home=B, P(B wins)=0.6 . . . "*

**Initial memento** (pass 1, score 5/10): Defines target probability P[(A,B)=(3,3) after 6]. Home team alternates each game; home win prob=0.6. Proposes DP approach and considers enumeration of all sequences.

**Judge feedback:** *Missing formula: recurrence f(n,a,b) with starting condition f(0,0,0)=1. Missing: explicit transition probabilities for odd/even games. Replace "proposes DP approach" with the named state variables and recurrence.*

**Refined memento** (pass 2, score 8/10): Defines f(n,a,b)=P(A has a wins, B has b wins after n games); f(0,0,0)=1; target f(6,3,3). Home pattern: odd games home=A, even home=B; p homewin=0.6. Transitions: if home=A, f(n,a,b)+=f(n-1,a-1,b)\*0.6+f(n-1,a,b-1)\*0.4; if home=B, swap.

Figure 10: Iterative memento refinement on a reasoning block about NBA playoff probabilities. The initial memento (pass 1, score 5/10) describes the *approach* but omits critical formulas. After judge feedback requesting the specific recurrence and transition probabilities, the refined memento (pass 2, score 8/10) captures the full computational state: function definition, base case, target, and recurrence relation.

## **A.2 Training and Evaluation Details**

### <span id="page-19-1"></span>**A.2.1 Training Details**

This section provides full details of the SFT training pipeline, complementing the high-level description in Section [4.](#page-5-1)

**Training Configuration and Hyperparameters.** All SFT experiments use TRL's SFTTrainer [\(von Werra](#page-16-14) [et al.,](#page-16-14) [2020\)](#page-16-14) with PyTorch 2.8+ and its native SDPA attention backend. All models are trained on 31K samples from OPENMEMENTOS with 32K sequence length on 32 NVIDIA B200 GPUs (4 nodes × 8 GPUs, 192 GB HBM per GPU). All runs share the same hyperparameters: AdamW optimizer (*β*1=0.9, *β*2=0.999, no weight decay), learning rate 8 × 10−<sup>5</sup> with cosine schedule and 5% warmup, 5 epochs per stage, gradient clipping at 1.0, global batch size 512, bfloat16 precision, gradient checkpointing, and seed 42. Checkpoints are saved every 50 steps (100 for Qwen3-32B). Qwen3-8B and Olmo-3-7B fit without model sharding; Phi-4 uses DeepSpeed ZeRO-2; Qwen3-32B requires ZeRO-3.

## **Two-Stage Training Procedure.**

**Stage 1 (Full Attention).** The environment variable KEEP LAST N BLOCKS=-1 disables block masking. Loss is computed on all completion tokens (prompt tokens are masked via labels=-100). Checkpoints are saved at regular intervals and evaluated on AIME24 to select the best checkpoint for Stage 2.

**Stage 2 (Memento Attention).** The environment variable KEEP LAST N BLOCKS is set to 0. The model is initialized from the best Stage 1 checkpoint and trained with identical hyperparameters. Loss is computed on all tokens, identical to Stage 1. The only difference from Stage 1 is the attention pattern: the custom block-masked attention implementation (Section [A.2.1\)](#page-20-0) applies a sparse attention mask during the forward pass, ensuring that tokens after a completed block+summary cannot attend to the masked block content.

<span id="page-20-0"></span>**Sparse Attention Mask Implementation.** We maintain custom model forks for each architecture family (Qwen3, Phi-4, Olmo 3) that modify the forward() method to support block masking during both training and inference. The implementation works as follows:

- 1. A **block cache** tracks the position type of every token: BLOCK, SUMMARY, or OTHER. It detects the learned special tokens (<|block start|>, <|block end|>, <|summary start|>, <|summary end|>) in the token stream.
- 2. When <|summary end|> is generated, the block cache marks the preceding thinking block as **completed**. All KV-cache entries for that block's reasoning tokens are masked from future queries.
- 3. The block cache is **stateful across autoregressive steps**, persisting through the KV cache. This avoids re-scanning the full sequence at every generation step.
- 4. For **training**, the full sequence is available, so the attention mask is constructed upfront as a dense mask matrix that zeroes out attention from post-summary tokens to their corresponding block content.

**Special Token Initialization.** Four special tokens are added to the tokenizer vocabulary: <|block start|>, <|block end|>, <|summary start|>, <|summary end|>. Their embeddings are initialized as the mean of semantically related existing tokens plus small Gaussian noise (*σ*=0.01):

- <|block start|> ← mean(*block*, *start*, *begin*, *section*, *step*)
- <|block end|> ← mean(*block*, *end*, *finish*, *section*, *done*)
- <|summary start|> ← mean(*summary*, *summarize*, *brief*, *recap*, *overview*)
- <|summary end|> ← mean(*summary*, *end*, *finish*, *done*, *complete*)

**Data Format.** Training data is pre-tokenized into HuggingFace Arrow format for efficiency. Each example is formatted in ChatML:

- **User message:** the problem statement.
- **Assistant response:** <think> + [<|block start|> reasoning <|block end|> <|summary start|> summary <|summary end|>] <sup>∗</sup> + </think> + final answer

For Qwen3 models, no system prompt is used (matching Qwen3's default behavior). For Phi-4, the native system prompt is preserved. Sequences are tokenized to a maximum length of 32,768 tokens with truncation; no padding is applied during tokenization (the data collator handles dynamic padding at batch time).

