## Expert Swap via Row-Swap Buffer (基于行交换缓冲区的专家交换)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Swap via Row-Swap Buffer 是 Stratum 中执行 Mono3D DRAM 内部 tier-to-tier expert 参数迁移的硬件机制。当 serving scheduler 从一种 topic batch 切换到另一种 topic 时（如从 "math" 切换到 "code"），不同 topic 对应的 hot expert 集合不同，需要将原来在快 tier 的 expert（现已成为 cold）与原来在慢 tier 的 expert（现已成为 hot）交换物理存储位置。传统方法需通过 xPU→interposer→DRAM→interposer→xPU 路径进行数据搬移，延迟和能耗极高。Stratum 在每 PE 的 local memory controller 中内置 8KB row-swap buffer，支持同一 DRAM bank 内的 row-to-row data swap，无需 traversing interposer 瓶颈接口。交换时间 <0.37%，能量 <0.03‰。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Expert Swap between consecutive topic batches (in NMP mode)

# Before batch dispatch:
Input: old_topic (e.g., "math"), new_topic (e.g., "code")
       old_placement P_old, target_placement P_target from Algorithm 1

For each DRAM bank with conflicting placements:
  # Identify experts that need to swap tiers
  swaps = []
  For expert e in all experts:
    if P_old[e] in fast_tier and P_target[e] in slow_tier:
      # Was hot, now cold → evict to slow tier
      swaps.append((e, "evict"))
    elif P_old[e] in slow_tier and P_target[e] in fast_tier:
      # Was cold, now hot → promote to fast tier
      swaps.append((e, "promote"))
  
  # Execute swaps pair-wise (evict + promote = swap partners)
  For (e_cold, e_hot) in swap_pairs:
    # Step 1: Read row[cold_tier_addr] → Row-Swap Buffer
    local_mem_ctrl.read_row(cold_tier_addr, row_swap_buffer)  # 8KB per PE
    
    # Step 2: Read row[hot_tier_addr] → overwrite cold_tier_addr
    local_mem_ctrl.read_and_write(hot_tier_addr, cold_tier_addr)
    
    # Step 3: Write Row-Swap Buffer → hot_tier_addr  
    local_mem_ctrl.write_row(row_swap_buffer, hot_tier_addr)
    
    # Timing: each swap ~ tRC(cold_tier) + tRC(hot_tier) 
    # + Row-Swap Buffer access (negligible vs DRAM timing)
    # All within same bank → no cross-bank movement needed

# Per-benchmark overhead (Table 4):
# OLMoE: 5.91 swaps/sec, 0.64ms (0.37%), 0.25mJ (<0.02%)
# Mixtral: 2.59 swaps/sec, 0.90ms (0.23%), 0.35mJ (<0.03‰)
# Llama-4: 4.02 swaps/sec, 0.45ms (0.18%), 0.34mJ (<0.02‰)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Row-swap buffer 的关键设计考量：(1) 仅在 bank 内执行 swap（同一 bank 的快 tier 和慢 tier 区间），充分利用 Mono3D DRAM 的内部带宽（19-34 TB/s），避免穿越 interposer；(2) 8KB buffer 大小足够容纳一个 row-buffer 页面（32Kb = 4KB），double-buffered 支持连续 swap；(3) 由 local memory controller 的 programmable state machine 控制，无需 xPU 参与（由 scheduler 通过 command queue 触发）；(4) 仅使用 dedicated buffer（非 shared memory），避免与 tensor core 计算争抢 SRAM 带宽。主要局限：仅支持 bank 内交换（同一 bank 不同地址），若需跨 bank 移动 expert（tensor-parallel sharding 变化），需 traversing interposer；但 Stratum 的 tensor parallelism 策略保持 expert shard placement 在 banks 之间不变，仅调整同一 bank 内的 tier 归属。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving
