## DSA Memory Move 运行时操作与延迟信号（time-slot 调制 + 中位数阈值解码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Move 是 Intel DSA 最基础的数据搬移操作（一次读 + 一次写），由 64B descriptor 指定源/目标地址与长度提交到 WQ、engine 异步执行。DarkStream 把它当作运行时信号原语：在 9 种 Source×Sink 操作组合（Memory Move / Fill / Compare Pattern）系统测量中，Move-Move 产生最大且最稳定的争用延迟差（Move 双向搬移、负载最重；Fill 单写、延迟最短但信号弱；Compare 依赖 DRAM 读、带宽利用低），因此选 Memory Move 作为隐蔽信道与侧信道的统一原语。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
隐蔽信道运行时循环（Source 与 Sink 分属不同 CPU core、不同 group 的 DWQ+engine，仅共享 DSA 设备）：

```
# Source（发送方）：每 time slot 编码 1 bit
for bit in msg:                      # 128-bit 帧 + 10101010b 前导码
    if bit == 1:
        while slot_not_over:         # active 态：异步饱和提交
            enqcmd(movemove_desc(1B))
    else:
        sleep(slot)                  # idle 态：不提交
# Sink（接收方）：持续提交并逐请求计时
latencies = []
while True:
    t0 = rdtsc(); enqcmd(move(1B)); wait_completion(); t1 = rdtsc()
    latencies.append(t1 - t0)        # idle≈1400 cycles, active 2000-4000 cycles
# 解码：每 slot 取中位数与阈值比较
for slot in slots:
    bit = 1 if median(latencies[slot]) > threshold else 0
```

侧信道变体：攻击者持续提交 1 MB Memory Move——大传输延长共享数据通路占用时间，使受害者干扰在长操作内累积放大（1 B/1 KB 探测下延迟方差大、无法区分受害传输尺寸，1 MB 下延迟与受害传输尺寸强正相关），逐请求记录延迟得到指纹 trace。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现 = 用户态进程经 Linux idxd 驱动 write syscall / ENQCMD 提交 descriptor，无需 root（只要管理员分配了 WQ 权限）。带宽 = 信道容量 × 传输频率，容量按 Binary Asymmetric Channel 模型由 0→1/1→0 bit-flip 概率 ε0/ε1 计算（式见论文 Eq.2），传输频率即 slot 时长倒数。扫描 40–256 KHz：低频容量≈1，147 KHz 处带宽峰值 129 Kbps，之后容量下降主导、带宽回落；128-bit 帧 + 8-bit 前导码同步。抗干扰：CPU 90% 负载下仍 >100 Kbps（DSA 执行与 CPU 解耦）；1 MB 静态 DSA 噪声下约 70 Kbps、随机噪声（每操作随机选 4 KB/64 KB/1 MB）下至多 49 Kbps。跨处理器：双路 Xeon Gold 6554S 上 Local-Local 92 Kbps、Local-Remote 78 Kbps、Remote-Local 92 Kbps。

涉及论文标题：
- DarkStream: Exploiting Internal Throughput Contention in Data Streaming Accelerator for Timing Attacks
