## gpt-oss / gpt-oss 120B

术语是什么？
gpt-oss是OpenAI发布的open-source大型语言模型系列（论文引用[65]: OpenAI 2025 "Introducing GPT-OSS"）。gpt-oss 120B是120B参数的Mixture-of-Experts (MoE)架构版本，基于Llama-style架构。模型使用FP4精度（4-bit浮点权重），128个experts（MoE routing，每次激活top-k个experts），hidden size为2,880，36层transformer blocks。该模型在HNLPU论文中被选为系统级评估的target model，是论文"极致专用化"论点的基础——一个主导性的预训练LLM作为通用认知基底，使hardwired实现变得合理。

从算法pipeline角度拆解：
gpt-oss 120B在HNLPU中的推理pipeline（Token-In-Token-Out，36层transformer每层）：
```
1. Input: token embedding (1, 2880) from HBM
2. For layer = 1 to 36:
   a. GQA Attention:
      - Q = X × Wq  (column-wise partitioned, all-reduce)
      - K = X × Wk  (column-wise partitioned, reduce→chip#)
      - V = X × Wv  (same as K)
      - Attention = Softmax(Q×K^T/√d) × V  (VEX FlashAttention)
      - Xo = Attention × Wo + X  (row-wise Wo, all-reduce + all-gather)
   b. MoE FFN:
      - Xnorm = RMSNorm(Xo)
      - Xrout = Xnorm × Wrout (router: Wrout (2880,128), top-k)
      - Xup = masked_X × Wup (8 experts/chip, 128 experts total, parallel)
      - Xgate = masked_X × Wgate (same partitioning)
      - Y = SwiGLU(Xgate) ⊙ Xup × Wdown + Xo (all-chip all-reduce)
3. Unembedding: Y × Wue → logits → Sampling → output token ID
```
模型配置：hidden size 2880, 64 query heads (GQA: 每8 query heads对应1 KV head), 128 experts (每chip 8 experts), FP4 weight精度。

术语一般如何实现？如何使用？
模型在GPU baseline（H100）上通过TensorRT-LLM部署；在HNLPU中，模型权重被物理固化到芯片金属导线中——不需要任何软件框架加载和运行。gpt-oss 120B的FP4量化是OpenAI原生支持的，论文未进行额外压缩。模型作为通用认知基底：用户通过prompt（自然语言token序列）而非ISA指令来编程hardwired处理器，利用in-context learning和zero-shot reasoning执行任意任务。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

