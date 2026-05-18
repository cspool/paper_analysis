## Dynamic PIM Command Scheduling (DCS)

术语是什么？通过联网搜索让回答具体和精准。

Dynamic PIM Command Scheduling（DCS）是PIMphony提出的PIM controller增强机制，将传统PIM的静态固定顺序命令发射替换为基于entry-level data dependency的乱序发射。传统PIM controller按固定时间间隔（tWR-INP、tMAC、tRD-OUT）串行发射WR-INP→MAC→RD-OUT命令序列，即使命令间无真实hazard也等待保守间隔，导致MAC pipeline大量idle。DCS在PIM HUB controller中增加Dependency Table（D-Table，记录每个GBuf/OBuf entry的最近访问命令ID）、Status Table（S-Table，记录每个命令的ID、完成时间和OBuf的is-MAC flag）和dependency-check unit。新命令到达时，controller查询D-Table/S-Table——仅当命令依赖的前序命令未完成时才等待，否则立即乱序发射，实现I/O数据搬运与MAC计算的overlap。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// DCS Scheduling Logic (per command):
function dcs_schedule(cmd):
    // cmd has: type (WR_INP/MAC/RD_OUT),
    //           gbuf_entry (for WR_INP/MAC),
    //           obuf_entry (for MAC/RD_OUT)

    if cmd.type == WR_INP:
        // Check: any pending MAC reading same GBuf entry?
        pending = d_table[cmd.gbuf_entry].has_pending_read()
        if not pending:
            issue_now(cmd)
        else:
            wait_for(d_table[cmd.gbuf_entry].last_read_cmd)

    if cmd.type == MAC:
        // Check: WR_INP to same GBuf entry done?
        wr_done = s_table[d_table[cmd.gbuf_entry].last_write_cmd].completed
        // Check: previous MAC to same OBuf entry done?
        obuf_free = not s_table.has_pending_mac_to(cmd.obuf_entry)
        if wr_done and obuf_free:
            issue_now(cmd)
        else:
            wait_for(max(wr_done_cmd, obuf_mac_cmd))

    if cmd.type == RD_OUT:
        // Check: MAC writing to same OBuf entry done?
        mac_done = s_table[d_table[cmd.obuf_entry].last_mac_cmd].completed
        if mac_done:
            issue_now(cmd)
        else:
            wait_for(mac_done_cmd)

// 静态vs动态调度示例（FP16 GEMV）:
// 静态: W0→W1→W2→M3→M4→M5→R6（串行，34 cycles）
// 动态: W0→W1→M3(等W0done)→W2(与M3并行)→M4(等W1done)
//       →R6(等M3done)→M5(等W2done)→M7(与R6无冲突,提前)
//       （乱序重叠，22 cycles）
```

DCS的关键使能硬件：(1) dual-port OBuf——port A被MAC写入时port B可同时读出已完成结果（或反之），允许MAC和RD-OUT在不同OBuf entry上并行；(2) multi-entry GBuf——允许WR-INP预取下一批数据到其他GBuf entry，MAC消费当前entry时不受影响。在GQA row-reuse场景下，DCS利用dual-port在MAC消费当前GBuf query entry时预取下一批query/score，将row-reuse的KV复用转化为真实吞吐收益。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DCS在PIM HUB controller中以硬件逻辑实现：D-Table和S-Table为per-controller 576B metadata SRAM，dependency-check unit为组合逻辑（比较当前命令的operand entry与table中记录的最近命令ID/状态）。Compiler在生成PIM指令时嵌入dependency annotations（标识每条指令读/写哪个GBuf/OBuf entry），runtime DCS controller在issue前做轻量级查表和比较。Paper对比ping-pong buffering baseline：ping-pong因静态调度不知道entry级依赖，需等两个region均idle才能切换（hand-off pipeline stalls），DCS以同buffer size实现up to 1.4× higher compute-unit utilization。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

