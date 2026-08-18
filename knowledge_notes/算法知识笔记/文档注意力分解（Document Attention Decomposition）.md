## 文档注意力分解（Document Attention Decomposition）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MERIDIAN（ISCA'26，HUST）提出的去中心化 RAG 推理机制：把标准 softmax 注意力模块按数据来源拆成两个独立分支——DocumentAttention 分支（对预计算的文档侧 K/V 做注意力）与 QueryResponseAttention 分支（对用户 query 与已生成 token 的 KV 做注意力），各自只产出紧凑局部摘要（未归一化输出 o、局部最大值 m、归一化因子 l），再经数值稳定的 online-softmax 全局融合（共享基线 m=max(m_d,m_c)）合并。与一般"注意力矩阵按 token 类别结构化拆分"（如 VLM KV Cache 剪枝里的 Intra/Inter-modality attention decomposition，见知识库笔记）不同，MERIDIAN 的分解是执行范式的改变：文档侧 K/V 按 attention head 分片静止在 PIM 内存设备上，每个设备对本地 shard 就地算注意力，只交换紧凑统计量，无需把整份文档 KV 搬到计算设备。数学上等价于标准 softmax（精确无损），下游 LN/FFN/残差全部不变。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MERIDIAN 算法 1（In-Layer Document Attention Decomposition）逐 token 流程（单设备持文档 KV (K_d,V_d)、另一设备持上下文 KV (K_c,V_c)）：
```
# 1) QKV 投影
(q, k, v) = QKVProjection(x)
# 2) 文档分支（PIM 设备本地执行）与上下文分支并行：
s_d = q @ K_d^T ;  s_c = q @ K_c^T          # 局部 logits（GEMV）
m_d = max(s_d) ;   m_c = max(s_c)           # 局部 max 基线
o_d = Σ_j exp(s_d[j]-m_d)·V_d[j] ;  l_d = Σ_j exp(s_d[j]-m_d)
o_c = Σ_j exp(s_c[j]-m_c)·V_c[j] ;  l_c = Σ_j exp(s_c[j]-m_c)
# 3) 全局融合（共享基线 m = max(m_d, m_c)，数值稳定）：
l = exp(m_d-m)·l_d + exp(m_c-m)·l_c
o = ( exp(m_d-m)·o_d + exp(m_c-m)·o_c ) / l
# 4) 下游不变：x ← LN1(x+o)；f ← FFN(x)；y ← LN2(x+f)
```
通信量对比（FP16）：集中式 V_ce = #Doc tokens×2×d_model×2 bytes；MERIDIAN V_de ≈ (#Query+#Response tokens)×2×d_model×2 bytes——文档平均比 query+response 长 ~380×（2Wiki/HQA/NQ/TQA 实测 doc 857–14749 token vs query+response ~20 token），通信降两个数量级以上；按 head 分片到 N 设备后每设备只传 d_model/N 输出切片，跨设备总流量近似恒定。跨设备融合与 Tree Attention 的 LSE/max 树归约同源（LogSumExp 结合律），但 MERIDIAN 在 PIM 设备粒度做两步（局部 → 全局）归约。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：与 PIM 加速器协同设计——文档 K/V 离线预计算后经 CXL.mem load/store 写入 head-sharded PIM 位置，DAC（Document Attention Cluster）执行文档分支、CEC（Context Execution Cluster）执行上下文分支与融合及全部其余算子；融合在 CEC 或 BOOMv2 RISC-V 核上完成（softmax 用专用精度硬件保证模型保真）。使用方式：KV-precomputed RAG 推理场景（TurboRAG/BlockAttention 微调过的模型），文档更新走标准 KV 预计算流程写对应 shard，无需全局重排。效果：MERIDIAN 通信占比 ≤6.34%（baseline 最高 93.40%），吞吐 5.36×/6.64×（vs TurboRAG/BlockAttention），准确率差 <0.4pp（LUT 近似仅用于数值宽容算子，softmax 专用精度）。

涉及论文标题：
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
