## Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation

- 属于算法pipeline的实现是什么？实验比较什么？
  - 近似匹配（论文本体为硬件架构，此处仅覆盖其"近似激活函数计算 + 精度保证"部分）：实现 = BF16 低精度 + Taylor 截断指数（n=4..7，e^x ≈ 1+x+…+x^n/n!）与 Newton 迭代开方替代精确 exp/sqrt 的近似激活计算 pipeline：Curry ALU 以 ArgReg=6 为迭代轮计数器，从最内层向外每轮执行 *=X、/=IterRound、+=1 直到 IterRound=0，每通道 16 bank × 2 路 = 32 路并发；sqrt 同法（Newton 迭代）。实验比较：Llama2-7B perplexity——FP32 基线 vs 原生 BF16 vs BF16 泰勒截断 n=4/5/6/7，三档长度（prefill 73/341/1139 + decode 15/65/270 tokens）：相对偏差 ≤0.3%（最显著 medium 档 n=5..7 相对 FP32 为 −0.251%），且误差不随上下文增长而累积。
- 硬件平台是什么，配置是什么。
  - 无 GPU 实测：perplexity 与全部性能评估均跑在 CompAir cycle-accurate 模拟器（ramulator2.0 + Booksim + CENT 模拟器 + SRAM-PIM 规格 [14]）上；近似 exp 的硬件开销由 Synopsys Design Compiler + UMC 28nm 综合（4×Curry ALU 的资源少于一个定制 16 输入 Softmax 单元，Vivado 对比）。
- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama2-7B（perplexity 主评估），其余端到端性能评估覆盖 Llama2-7B/13B/70B、Llama3（GQA 场景）、Qwen-72B、GPT3-175B。数据集：论文未明确说明 perplexity 所用具体数据集名称（仅给出 short/medium/long 三档 prefill/decode 长度）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Man0xbfc00380/comp-air.git（已确认，含 NoC 计算模型与 translate/ 翻译器）。算法 pipeline 执行例子（16 bank 上的 Softmax 分子计算）：① 每 bank 对本地分片 x_j 发 exp packet，flit Data=16b BF16；② Curry ALU 配置 ArgReg=6（迭代轮）、IterArg=1、IterOp='-'：每轮 ArgReg 先 −=1 再执行 acc = acc×X/IterRound + 1（自最内层向外展开 Taylor 截断），IterTag 触发 ArgReg 动态更新；③ 计算结果就地替换 flit Data 继续路由；④ NoC_Reduce 以 4 层二叉树对 16 bank 的 exp 结果求 Softmax 分母，广播回各 bank。效果：perplexity 相对 FP32/原生 BF16 偏差 <0.3%、不随 1139-token 长上下文累积；硬件代价为零额外流水级（计算与 switch traversal 并行），支持 32 路并发指数计算。
