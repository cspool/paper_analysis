## LLM 非线性算子（Softmax / SiLU / RMSNorm）与超越函数近似（Taylor / Newton 迭代）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Transformer 每层的非线性算子：Softmax（attention 权重归一，含 exp 与跨序列归约，延迟随序列长度线性增长）、SiLU（FFN 激活，sigmoid 系）、RMSNorm（RMS 归一，含 sqrt）。这些算子没有直接硬件原语，数字电路通常用迭代法近似：exp 用 Taylor 展开截断（e^x ≈ 1+x+x²/2!+…+x^n/n!）、sqrt 用 Newton 迭代。CompAir 实测这些非线性不可忽略：4K 上下文时占 block 时间约 20%、长上下文时通信+计算可超 25% 总延迟——推翻"非线性可省略"的假设，成为 CompAir-NoC 的核心动机。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CompAir 的近似 pipeline（Curry ALU、BF16）：
```
# exp 的 Taylor 截断（自最内层向外），ArgReg 作迭代计数器
ArgReg, IterArg, IterOp = 6, 1, '-'
acc = 1
while ArgReg > 0:
    acc = acc * X / ArgReg + 1   # *=X, /=IterRound, +=1
    ArgReg -= 1                  # IterTag 触发
# sqrt 同理 Newton 迭代；每通道 16 bank × 2 ALU = 32 路并发
```
精度验证（Table IV，Llama2-7B perplexity）：FP32 vs 原生 BF16 vs Taylor n=4/5/6/7，三档上下文（prefill 73/341/1139 + decode 15/65/270 tokens）：相对偏差 <0.3%（最显著 medium 档 n=5..7 相对 FP32 为 −0.251%），误差不随上下文增长累积。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：LUT + 多项式、CORDIC、迭代法；GPU 上以 exp2f/__expf 等近似指令；PIM 上用流式迭代（Curry ALU，零额外流水级）或专用 NLU（CENT 的 NLU 为 7nm 4.4mm²，约 4× 一个 32MB bank）。使用要点：迭代轮数决定精度-延迟权衡（n=4..7 均已验证可接受）；sqrt 的 Newton 迭代需初值估计与除法支持（Curry ALU 每 ALU 含 1 个 divider）；归约与 exp 在途合流避免中间结果搬移。

QiMeng-Tensify 补充视角（ISCA'26）：SiLU 在图级编译中的优化——GatedMLP 的 SiLU 由 exp/add/div/mul 四个子算子组成，初始为独立 kernel + 中间 buffer；QiMeng-Tensify 的 LLM 识别 SiLU(x)=x/(1+e^(-x)) 可融合表达式后，用 AutoInline 把四个子算子折叠为单一 SiLU block，再 compute_at 把 SiLU 提升进 GEMM 的 reduction 循环（S1→S4→S5），完全消除中间 buffer 与 kernel launch 开销。这体现非线性算子在编译框架层的"融合价值"：不止是近似精度问题，还有"可折叠进 GEMM 循环体避免物化中间张量"的访存优化。同样，RMSNorm/LayerNorm 子图（RMSNorm-LLaMA2-7B、LayerNorm-Transformer、nTrans 的 Norm+residual 融合）被用作 benchmark（Table VII），QiMeng-Tensify 通过布局处理与融合策略优于 TVM/Triton。注：本论文非线性算子不做数值近似（与 CompAir 的 Taylor/Newton 不同），其优化是"算子融合与调度"层面的。

涉及论文标题：
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
