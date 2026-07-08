# 知识库_kernel调度

## Expert Parallelism (EP)

术语是什么？
Expert Parallelism (EP) 是 MoE 模型的分布式训练/推理策略之一。将不同 expert 的 FFN 参数分布到不同 GPU 上，每个 GPU 持有部分 expert。每个 token 根据 Gate Network 的路由决策，通过 Dispatch 通信算子发送到持有其激活 expert 的远程 GPU 上执行计算，完成后通过 Combine 通信算子将结果聚合回 token 原始所在 GPU。EP 与 Data Parallelism (DP)、Tensor Parallelism (TP)、Pipeline Parallelism (PP) 共同构成 MoE 的 4D 并行策略。

从kernel调度角度拆解：
EP 的 kernel 执行流程（以单个 MoE layer 为例）：

```
// Step 1: Gate Network 在本地 GPU 执行
gating_result = gate_network_cuda_kernel(hidden_states)

// Step 2: Dispatch kernel — 将 token 发送到 expert GPU
for each token t in local_tokens:
    for each activated expert e in topk_indices[t]:
        target_gpu = expert_to_gpu_mapping[e]
        if target_gpu != local_gpu:
            // 发送 token hidden state 到 remote GPU
            send_token(t, target_gpu, expert_idx=e)
        else:
            local_expert_queue[e].append(t)

// Step 3: Expert Computation — 各 GPU 独立执行
for each expert e on this GPU:
    tokens = expert_queue[e]
    output_1 = GEMM_1(tokens, W_in[e])    // hidden -> intermediate
    output_2 = Activation(output_1)
    output_3 = GEMM_2(output_2, W_out[e])  // intermediate -> hidden
    // Multiply gating weight in GEMM-2 epilogue
    expert_outputs[e] = output_3 * gate_weight

// Step 4: Combine kernel — 聚合 expert 输出回源 GPU
for each token t:
    for each activated expert e in topk_indices[t]:
        source_gpu = expert_to_gpu_mapping[e]
        if source_gpu != local_gpu:
            recv_and_reduce(t, source_gpu)  // 累加 expert 输出
        else:
            local_reduce(t, expert_outputs[e])
```

术语一般如何实现？如何使用？
- DeepEP (DeepSeek) 是 SOTA EP 通信库，提供高度优化的 Dispatch 和 Combine kernel，支持 NVLink + RDMA  hierarchical 通信。
- NVIDIA Megatron-Core 的 Hybrid-EP 支持 FP8 通信和 computation-communication overlap。
- 关键优化方向：(1) 通信-计算重叠（FasterMoE/Tutel/COMET），(2) 冗余通信消除（DySHARP 通过 in-switch computing 消除 dispatch 多播冗余和 combine 归约冗余），(3) hierarchical 通信（intra-node NVLink + inter-node RDMA），(4) 融合 kernel（token-centric kernel fusion）。
- EP 通信占 MoE 层执行时间的 50-80%（DeepSeek-V3 上为 70.4%），是 MoE 训练的主要瓶颈。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch

## Dispatch / Combine（MoE 通信算子）

术语是什么？
Dispatch 和 Combine 是 Expert Parallelism 中的两个核心通信算子。Dispatch 负责将 token 从当前所在 GPU（通常为 token 原始所在 rank）发送到持有其激活 expert 的 GPU 上。Combine 负责将各 expert GPU 计算出的 token 输出聚合（加权求和）回 token 原始所在的 GPU。两者共同构成 MoE 中 all-to-all 通信模式，是 Expert Parallelism 的主要性能瓶颈。

从kernel调度角度拆解：
Dispatch 和 Combine 的 kernel 实现包含以下关键操作：

```
// Dispatch Kernel (per GPU)
__global__ void dispatch_kernel(
    Token* tokens, int* topk_indices, float* topk_weights,
    int* expert_to_gpu_map, int* send_counts, ...)
{
    // 1. 计算每个 token 的目标 GPU
    // 2. 按目标 GPU 排序/分组 token（保证连续内存访问）
    // 3. 打包 token hidden state + metadata（token ID, expert ID, weight）
    // 4. 通过 NVLink/RDMA 发送到目标 GPU
    //    - DeepEP: 使用 NVLink put/get 或 IB send/recv
    //    - DySHARP: 使用 dymultimem.st 指令，单次发送 + switch 多播
    // 5. 发送 metadata（token arrival count）通知目标 GPU
}

// Combine Kernel (per GPU)
__global__ void combine_kernel(
    Token* expert_outputs, int* token_ids, float* topk_weights,
    int* recv_counts, ...)
{
    // 1. 等待所有 expert 输出就绪（通过 token tracker 或软件 barrier）
    // 2. 从各 expert GPU 接收输出
    //    - DeepEP: 使用 NVLink get 或 IB recv
    //    - DySHARP: 使用 dymultimem.ld_reduce 指令，switch 内归约
    // 3. 执行加权求和: output += expert_output * gate_weight
    //    - DySHARP: 加权在 GEMM-2 epilogue 中完成，Combine 仅做加法归约
    // 4. 写回最终输出到 token 原始位置
}
```

术语一般如何实现？如何使用？
- DeepEP 使用高度优化的 CUDA kernel + NVLink 直接内存访问，支持 hierarchical intra-node + inter-node 通信。
- NVIDIA TensorRT-LLM 使用 one-sided all-to-all（单向 NVLink put/get），消除 send/recv 配对开销。
- DySHARP 的 dymultimem.st（替代 Dispatch）和 dymultimem.ld_reduce（替代 Combine）通过 in-switch computing 消除冗余传输：Dispatch 时 GPU→Switch 仅发一次，Switch 多播到所有目标 GPU；Combine 时 Switch 内归约后仅传回最终结果。
- 传统方案中 Combine 的加权求和通过软件实现（需传输 weight 元数据），DySHARP 将 weight 乘法移至 GEMM-2 epilogue 以避免在 switch 中实现加权归约（硬件复杂度高）。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch

## Token-Centric Kernel Fusion（Token 中心化 Kernel 融合）

术语是什么？
Token-Centric Kernel Fusion 是 DySHARP 提出的 kernel 调度技术。它将 MoE 层的 Dispatch-GEMM1-GEMM2-Combine 四个算子视为一个 token-paced pipeline（token 步调流水线），以 token/tile 粒度追踪数据依赖关系，一旦某个 tile 的输入就绪即立即调度执行其下游算子，而无需等待整个算子完成。这使得 Dispatch（GPU→Switch 为主）和 Combine（Switch→GPU 为主）可以并发执行，合并双向互补的非对称通信模式，从而将 in-switch computing 的流量减少转化为实际加速。

从kernel调度角度拆解：
Token-Centric Kernel Fusion 的核心调度逻辑：

```
// Token-Centric Scheduler (megakernel 内 persistent TB 实现)
while (true):
    // 1. Dispatch TB: 发射 dymultimem.st
    for each ready_token_tile:
        if SM_group[DISPATCH] has capacity:
            issue_dymultimem_st(tile)  // 更新 TS Table DAcc 字段
    
    // 2. Poll readiness: Dispatch -> GEMM-1
    for each tile_entry in TS_Table:
        if tile_entry.DAcc >= tsize * bsize:    // tsize tokens 全部到达
            mark GEMM1_TB_row[tile_entry.Row] as READY
    
    // 3. GEMM-1 TB: 就绪即发射
    for each ready_GEMM1_row:
        if SM_group[GEMM1] has capacity:
            issue_GEMM1_TB_row(row)  // 完成后更新 TS Table TBCnt1
    
    // 4. Poll readiness: GEMM-1 -> GEMM-2
    for each tile_entry in TS_Table:
        if tile_entry.TBCnt1 == total_TBs_in_row:  // 该行 GEMM-1 全部完成
            mark GEMM2_TB_row[tile_entry.Row] as READY
    
    // 5. GEMM-2 TB: 就绪即发射
    for each ready_GEMM2_row:
        if SM_group[GEMM2] has capacity:
            issue_GEMM2_TB_row(row)  // 完成后更新 TS Table TBCnt2
            // 完成后: 通知 source GPU 更新 OR Table
    
    // 6. Poll readiness: GEMM-2 -> Combine
    for each token_entry in OR_Table:
        if token_entry.nReady == topk:       // 所有 topk expert 输出就绪
            mark token as COMBINE_READY
    
    // 7. Combine: 就绪即发射 dymultimem.ld_reduce
    for each combine_ready_token:
        if SM_group[COMBINE] has capacity:
            issue_dymultimem_ld_reduce(token)
    
    if all_tokens_processed:
        break
```

SM 分区：SMs 分为四组（Dispatch/GEMM-1/GEMM-2/Combine），GEMM-1 和 GEMM-2 在无可调度 TB 时可共享 SM。

术语一般如何实现？如何使用？
- 通过 megakernel + persistent thread blocks 实现，绕过硬件 TB scheduler。
- 原始 TB 被表示为 "task"，persistent TB 从 task list 中取 task 执行。
- Readiness-gated schedule：kernel 通过专用 load 指令轮询 token tracker 表中就绪标志位（忙等循环）。
- 同步 tile size = 128（匹配 GEMM tile size），在计算利用率和重叠粒度间取得平衡。
- 必须与 dynamic multimem addressing 配合使用才有效果：kernel fusion alone 无法超过 SOTA baseline COMET（因为未消除通信冗余）；dynamic multimem addressing alone 无法将流量减少转化为加速（因双向非对称瓶颈）。
- 其他 MoE overlap 方案：FasterMoE/Tutel（粗粒度 overlap）、COMET/CCFuser（细粒度 overlap），但均将 Dispatch 和 Combine 作为独立 kernel 执行，无法合并双向互补通信。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch

## Megakernel / Persistent Thread Blocks

术语是什么？
Megakernel 是一种 GPU kernel 实现模式，将所有算子融合为单个持久化 kernel，使用 persistent thread blocks (TBs) 绕过 GPU 硬件 TB scheduler。原始 GPU kernel 的每个 TB 被转化为一个 "task"，persistent TB 在 SM 上持续运行，从 task list 中循环取 task 执行，实现软件级别的 TB 调度。这种模式允许细粒度的 readiness-gated 调度和跨算子同步。

从kernel调度角度拆解：
```
// Megakernel 结构
__global__ void moe_megakernel() {
    // Persistent TB: 持续运行直到所有 task 完成
    while (true) {
        Task task = fetch_next_task(task_list);  // 原子操作取 task
        if (task == NO_MORE_TASKS) break;
        
        switch (task.type) {
            case DISPATCH_TILE:
                // 发射 dymultimem.st
                dispatch_tile(task.tile_id);
                update_TS_Table_DAcc(task.tile_id);
                break;
            case GEMM1_TB:
                // 等待 Dispatch 就绪 (poll TS Table)
                while (!is_GEMM1_ready(task.row_id)) { /* spin */ }
                execute_GEMM1_TB(task.row_id);
                update_TS_Table_TBCnt1(task.row_id);
                break;
            case GEMM2_TB:
                while (!is_GEMM2_ready(task.row_id)) { /* spin */ }
                execute_GEMM2_TB(task.row_id);
                update_TS_Table_TBCnt2(task.row_id);
                notify_source_GPUs(task.token_ids);
                break;
            case COMBINE_TOKEN:
                while (!is_Combine_ready(task.token_id)) { /* spin */ }
                execute_dymultimem_ld_reduce(task.token_id);
                break;
        }
    }
}
```

术语一般如何实现？如何使用？
- 需要 SM 分区以分配不同算子类型的 TB 到不同 SM 组，避免资源竞争。
- 忙等轮询（spin-polling）通过专用 load 指令读 token tracker 硬件表，而非软件共享内存同步。
- 优势：消除 kernel launch overhead，支持更细粒度的算子间依赖追踪和调度。
- 局限性：所有算子必须在同一 megakernel 内实现，调试和维护复杂度高；SM 静态分区可能导致某些算子的 SM 利用率不足。
- Rammer (OSDI'20) 首先提出 rTask 概念和 persistent TB 调度。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch

## Token Tracker（硬件 Token 追踪器）

术语是什么？
Token Tracker 是 DySHARP 提出的硬件模块，用于在 token/tile 粒度上追踪 MoE 层 Dispatch→GEMM-1→GEMM-2→Combine 的数据依赖关系。它由三个表组成：(1) Tile Status (TS) Table — 追踪每个 tile 的 Dispatch 到达量和 GEMM TB 完成状态；(2) Token ID (TID) Table — 记录每个 tile 内的 token ID；(3) Output Readiness (OR) Table — 追踪每个 token 的 Combine 就绪状态。Token Tracker 是 token-centric kernel fusion 的关键基础设施，使 scheduler 可以在 readiness boundary 上调度。

从kernel调度角度拆解：
Token Tracker 的依赖追踪流程：

```
// TS Table Entry 字段:
//   Valid | ExpID | Row | DAcc | TBCnt1 | TBCnt2

// 1. Dispatch -> GEMM-1 readiness:
//    每到达一个 dymultimem.st（写入 bsize bytes）:
TS_Table[entry_idx].DAcc += bsize
//    当 DAcc == tsize * bsize 时: GEMM-1 TB row 就绪

// 2. GEMM-1 -> GEMM-2 readiness:
//    每完成一个 GEMM-1 TB:
TS_Table[entry_idx].TBCnt1 += 1
//    当 TBCnt1 == total_TBs_in_row 时: GEMM-2 TB row 就绪

// 3. GEMM-2 -> Combine readiness:
//    GEMM-2 TB row 完成后:
//    a) TS_Table 中 TBCnt2 == total_TBs_in_row
//    b) 读 TID_Table[TPtr] 获取该 tile 的所有 token ID
//    c) 向每个 token 的 source GPU 发送通知
//    d) Source GPU 的 OR_Table[token_id].nReady += 1
//    e) 当 nReady == topk 时: token Combine 就绪
```

调度器轮询就绪标志（DAcc/TBCnt/nReady）以决定何时发射下游算子。

术语一般如何实现？如何使用？
- TS Table 和 OR Table 驻留在 on-chip SRAM（各 1024 entry），溢出时 offload 到 DRAM。
- TID Table 驻留在 DRAM（因尺寸较大且访问频率低）。
- 所有状态更新在数据可见后执行（detect acknowledge 表明数据已到达 LLC/DRAM）。
- 16-bank dual-port SRAM (1R1W) 满足并发读写需求。
- 是 token-centric kernel fusion 的关键硬件基础设施，与传统软件 barrier 同步相比，提供了 token/tile 粒度的细粒度依赖追踪。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch
