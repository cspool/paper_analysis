## DGL（Deep Graph Library，深度图库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DGL 是面向图神经网络（GNN）的开源 Python 软件框架（Wang et al., 2019，"Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs"，https://github.com/dmlc/dgl），提供图数据结构（DGLGraph）、消息传递 API（update_all/message_func/reduce_func）、SpMM/SDDMM 等稀疏原语与对 PyTorch/TensorFlow 的后端集成。其核心抽象把 GNN 训练/推理建模为沿边的消息生成（SDDMM）与沿顶点的消息聚合（SpMM），底层依赖 MKL/cuSPARSE 等库。
- TAGT 论文中的角色：DGL 是通用 GT 软件 baseline 的载体——DGL-CPU（v2.4.0）保留 O(N²) 全局注意力跑在 32 核 Intel Xeon Platinum 8357B 上，作为全注意力精度参考；同时 TAGT-S 是"用 TDS 拓扑感知稀疏化与合并方法修改 DGL"的软件实现（跑 A100 GPU），用于在硬件之外验证 TDS 算法的收益。论文指出 DGL/PyG 主要优化稀疏 message-passing（SpMM 1-hop 聚合），不高效支持 GT 的稠密全对注意力。

从编译框架角度拆解术语，比如术语所在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 框架内 GT 执行流程（以 DGL 跑一个图 Transformer 层为例）：① 构建全连接 DGLGraph（每对顶点连边+自环）并绑定顶点特征与结构编码；② 用 DGL 的异构算子/图遍历取每个顶点的邻居特征矩阵 H^v；③ 在 PyTorch 后端做 Q/K/V 线性投影、QK^T 注意力、softmax、PV 聚合（全对交互，O(N²)）；④ FFN+残差后进入下一层。DGL-CPU 场景下上述矩阵运算落到 Xeon CPU（Intel MKL/DNNL），图遍历按 CSR 布局进行。
- TAGT-S 的修改点：用 TDS 稀疏子图（original/fusion/association 三类边，见 算法pipeline 层 TDS 条目）替换全连接邻接，使注意力只在其 1-hop 邻域（O(m·log_m N) 个顶点）上计算——即把 DGL 的图结构从全连接图换成 TDS，其余投影/softmax/FFN 计算沿用 DGL+PyTorch 路径。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 通用用法：`import dgl; g = dgl.graph((u, v)); g.ndata['feat'] = x; g.update_all(fn.u_mul_e('feat','w','m'), fn.sum('m','h'))` 式消息传递；配合 PyTorch 做端到端 GNN/GT 训练与推理。DGL v2.4.0 开源（https://github.com/dmlc/dgl）。
- TAGT 论文用法：作为 GT baseline 框架（DGL-CPU 全注意力参考）与 TDS 算法载体（TAGT-S）。注意 DGL 也用于其它图加速论文（如 TailorKV 用 DGL 稀疏 tensor API 直接行传输 KV、Token Condensation 用 DGL 构建 token 图）。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging
