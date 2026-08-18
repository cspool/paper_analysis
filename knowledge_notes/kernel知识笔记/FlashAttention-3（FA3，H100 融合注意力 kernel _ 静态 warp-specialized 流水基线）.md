## FlashAttention-3（FA3，H100 融合注意力 kernel / 静态 warp-specialized 流水基线）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FA3 是 H100/Hopper 上高度手调的融合注意力 kernel（论文：Shah et al., NeurIPS 2024，"Fast and Accurate Attention with Asynchrony and Low-precision"，arXiv:2407.08608），把 QK^T 乘、缩放/掩码、softmax、V 投影融合进单个 kernel，利用 Hopper 的 TMA（Tensor Memory Accelerator，硬件异步张量搬移）与 WGMMA（Warpgroup MMA，warpgroup 级异步张量核指令）实现生产者-消费者 warp 特化流水。FA2 在 H100 上只有 ~35% 理论利用率，FA3 通过异步重叠达到 FP16/BF16 约 740 TFLOPS（~75% 利用率）、FP8 近 1.2 PFLOPS，FP16/BF16 比 FA2 快 1.5–2.0×。
- 在本论文中的角色：作为"静态 tile 级流水调度"的代表性手调 kernel 基线。论文把 FA3 融合注意力分解为三类异构 tile：M0（QK^T GEMM，tensor 单元）、S（softmax，vector 单元）、M1（AV GEMM，tensor 单元），每迭代垂直一组 M0/S/M1，迭代间由静态屏障强制同步。FA3 依赖 8 处显式同步（warpgroup_fence_producer、wgmma::wait、warpgroup_barrier_arrive/wait、warpgroup_commit_batch 等）锁定静态双阶段/三阶段重叠模板，TMA 停顿会使依赖消费者序列无条件阻塞。
- Web 佐证：FA3 三个核心技术——生产者-消费者异步（warp 特化重叠数据搬移与计算）、异步块式 GEMM 下重叠 softmax（warpgroup 间 pingpong + warpgroup 内双级流水）、FP8 硬件加速 GEMM（块量化 + incoherent processing）。生产者 warps 发 TMA load 到循环共享内存 buffer 并写命名屏障，消费者 warpgroups 回收寄存器（setmaxnreg）等屏障后 WGMMA 计算。开源在 https://github.com/Dao-AILab/flash-attention（hopper/ 目录）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- CUDA 版调度过程（论文 Fig.5 左伪代码）：tma_load_q/tma_load_k_transpose 载入 s_Q/s_K → warpgroup_fence_producer → wgmma::mma_sync(s_P,s_Q,s_K)（M0）→ wgmma::wait + softmax_warpgroup(s_S,s_P,state)（S）→ 循环 j=0..Tc-1：tma_load_k_transpose(s_K_next) + warpgroup_barrier_arrive + wgmma::mma_async(s_S_next,s_Q,s_K_next)（预取下一 K tile）→ tma_load_v(s_V,V) + warpgroup_barrier_wait → wgmma::mma_sync(s_R,s_S,s_V)（M1）→ wgmma::wait + softmax_warpgroup(s_S_next,...) → wgmma::wait + rescale_warpgroup(s_O,s_R,state,state_next) → warpgroup_commit_batch + update_carousel_index → tma_store_o。要点：8 处显式同步把 S^i 与 M0^{i+1}（数据独立、ME/VE 资源不冲突）序列化，迭代间形成隐式屏障。
- TISA 版（Epoch）对比：同一循环以 tisa::gemm<me>/tisa::softmax<ve>/tisa::load<de>/tisa::store<de> 声明式表达、零屏障；硬件调度器按依赖就绪乱序发射，S_i 与 M0_{i+1}、M1_i 与 S_{i+1} 并发（图 2c/e）。TISA kernel 编译器自动生成：代码量 -30%、同步频率 -50%、性能在手调基线 5% 内。
- 调度/重叠效果：静态双阶段（M0+S | M1）或三阶段（M0 | S | M1）模板只能节省固定量（E0/E1），动态调度实现更紧凑跨迭代重叠（E0+E2 或 E1+E3）。Accumulated Overlap Score 中 DMV 三单元同时激活类重叠在静态下为 0（表 VII）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：FA3 开源（https://github.com/Dao-AILab/flash-attention，Hopper SM90 实现，新版本用 CuTe DSL），手调 CUDA/CUTLASS 风格：thread-block 分解 + 共享内存 staging + warp 级 collectives + 手调预取；生产者/消费者 warp 特化 + 命名屏障 + pingpong 调度。论文将其作为 H100 基线原样运行。
- 实验用法：论文对比 Epoch(TISA FA3, BF16) vs H100(FA3) 的持续 BF16 吞吐，seq 长度 512–16K、带/不带 causal mask；硬件利用率 = Achieved GFLOPs / Peak GFLOPs；Epoch 在向量:矩阵计算比 1:8（H100 原生比）下全序列长利用率高 >10%，head dim 128 主流配置高 26.4%，1:16 比仍高 15.7%，1:32 比多配置相当——尽管 Epoch 带宽仅 H100 的 1/3.35（1.0 vs 3.35 TB/s），证明增益来自调度（TISA 消除静态 per-iteration 同步）而非算力。
- 结论性用法：作为"静态同步固定的 SOTA 手调 kernel"与"动态调度 kernel"的对照锚点，说明静态 barrier 无法适应运行时变动（TMA 停顿即阻塞依赖序列），而 TISA 按精确就绪轨迹调整发射序。

涉及论文标题：
- Dynamic Scheduling for AI Accelerators via TISA
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging（TAGT 把 FA 系列 IO-aware 注意力 kernel 作为对照：tiling+online softmax 只能降低稠密注意力的显存流量，不能消除 O(N²) 顶点对交互次数；且 GT 的结构编码使注意力矩阵不规则、块矩阵优化失效。TAGT 以 TDS 稀疏注意力 + FAU 流式分数/SCU 块级异步 softmax 替代）
