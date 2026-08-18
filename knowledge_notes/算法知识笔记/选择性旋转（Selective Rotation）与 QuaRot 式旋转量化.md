## 选择性旋转（Selective Rotation）与 QuaRot 式旋转量化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QuaRot（arXiv:2404.00456）利用计算不变性 Y=Wx=(WQ^T)(Qx)，用正交（Hadamard）旋转同时变换权重与激活以压制激活离群值，实现 W/A/KV 全 4-bit 量化；Hadamard 矩阵 H_2n=1/√2·[H_n H_n; H_n −H_n] 的相干性达到 Welch 界 μ=1/√n。QuaRot 定义四类旋转（R1-R4）：Q/K/V 投影与首个 MLP 投影的输入旋转（离线并入权重）、attention 输出投影旋转（离线）、attention Q/K 在线旋转（服务 KV 量化）、末层 MLP 投影输入在线旋转。PLENA 的"选择性旋转"是其 MX 格式适配变体：旋转只施加到增益为正的层子集 S（按每层 perplexity 增量 Δppl 搜索），权重不旋转（MX 小块共享指数已捕获权重离群，旋转反而增 PPL：MXINT4 权重 6.83→6.98），激活/KV 旋转在线执行并配 PLENA 硬件原生乘法支持（运行时乘 H^{-1}）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
l_rot*(X) = Q(X·H) · H^{-1} · Q(W)          # 只旋转激活，权重不旋转
S* = argmin_{s⊆M} Σ_{s∈M} Δppl(l_rot*)      # 按层搜索旋转子集（以 PPL 增量为准）
```
- KV 路径（PLENA）：新 K/V append 前做 Hadamard 旋转 → 量化为 MX 存 HBM → 读入 Matrix SRAM 后做逆 Hadamard 变换 → 进 attention GEMM；权重加载绕过逆变换（旋转/逆旋转可按张量选择性施加）。激活路径：XH 后量化、运行时乘 H^{-1} 复原（硬件 vector 单元提供旋转指令）。
- 消融结论（LLaMA-3-8B，Table VI）：激活/KV 量化中旋转有效——MXINT4 7.24→7.05、MXFP4 29.75→14.50（但 MXFP4 仍差于 MXINT4）；full-system 中 Erry 裁剪 7.60 + 选择性旋转 → 7.22。权重侧旋转则普遍有害。
- 与 QuaRot 的关键差异：QuaRot 把旋转并入权重（离线 R1-R4），PLENA 权重不旋转、激活/KV 旋转在线执行——因为 MX 小块共享指数的权重量化已能吸收权重离群，旋转的必要性只存在于动态范围宽的激活与 KV。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Hadamard 矩阵用块对角/Kronecker 结构（I⊗H_dh）低开销应用；旋转子集用逐层 PPL 增量启发式搜索；硬件侧把旋转做成向量单元指令、逆变换在数据加载路径上执行。使用：W/A/KV 全低比特量化时压制激活/KV 离群；注意与 RoPE 的兼容性（RotateKV 处理 RoPE 例外、用 outlier-aware 自适应旋转）。工具链：AMD Quark（ONNX/PyTorch）提供 QuaRot R1-R4 配置化实现、昇腾 MindStudio msModelSlim 将其作为离群值抑制算法；后继工作 SpinQuant（学习旋转矩阵）、ButterflyQuant（可学习蝶形变换，2-bit LLaMA-2-7B 上 PPL 15.4 vs QuaRot 22.1）、GSR（sequency 排序 Walsh 旋转）扩展了旋转族。

涉及论文标题：
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
