## Stage-Decoupled Scheduling (阶段解耦调度)

术语是什么？通过联网搜索让回答具体和精准。
Stage-Decoupled Scheduling是DFVG中实现FPGA draft和GPU verify阶段重叠执行的核心调度策略。与传统speculative decoding中draft→verify串行执行（或独立执行无协调）不同，Stage-Decoupled Scheduling将两阶段完全解耦：FPGA draft engine持续生成tokens不受verify结果影响（异步执行），GPU verify engine在收到draft后立即开始验证，同时FPGA继续生成下一轮drafts。两阶段通过execution window分配和rollback prediction协调，消除pipeline bubbles和同步等待开销。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Stage-Decoupled Scheduling的执行timeline：
```
Time →
FPGA: |Draft_1|Draft_2|Draft_3|Rollback|Draft_4|Draft_5|...
GPU:  |Idle|  |Verify_1|Verify_2+3|Forward_4|Verify_4|...
PCIe: |Tx1|  |Rx1|Tx2|Rx2|Tx3|...

关键解耦规则:
// FPGA侧:
while True:
    T = BuildTreeADAPT(prefix, M_draft)
    write_to_host_buffer(T)
    signal_ready()
    // 不等GPU verify结果，直接开始下一轮
    if GPU_returned:
        update_prefix_and_kv_cache()

// GPU侧:
while True:
    if host_buffer_ready():
        T = read_from_host_buffer()
        Tsorted = TreeSort-Verify(T)
        O = ParallelBlockAttention(Tsorted)
        accepted = AcceptTokens(T, O)
        return_accepted_to_FPGA()
    elif idle:
        forward_from_prefix()  // 主动forward避免bubble
```
关键特征：(1) FPGA永不休眠（持续drafting）→ 即使部分draft被rejected，算力不浪费（后续可能接受）；(2) GPU在无draft可verify时主动forward new tokens→避免idle bubble；(3) 解耦使两阶段独立scaling——FPGA频率和draft model size可不影响GPU verify throughput；(4) 异步通信使PCIe transfer与computation完全overlap。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DFVG以C++ host controller实现stage-decoupled scheduling：维护两个独立execution context（FPGA draft stream、GPU verify stream），通过shared memory flags和interrupt实现异步协调。FPGA侧使用custom compiler extension支持rollback recovery指令。配置参数：D_min = ⌈T_verify / T_draft⌉确保execution window足够重叠。该设计理念可推广至其他heterogeneous multi-device推理场景（如NPU+GPU、多芯片系统）。开源：https://github.com/ShaoqiangLu/DFVG。

涉及论文标题：
- DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU
