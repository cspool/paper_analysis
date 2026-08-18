## 预加密调度（Pre-encryption Scheduling / 机会性预加密）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
预加密调度是 LÆGIS 提出的运行时调度机制：在 GPU CC 的 UVM 故障批处理（fault batch handling）期间，把 AES-GCM 页加密这一运行时计算从关键路径卸载到 UVM driver 线程的空闲时段执行。前提是加密与访问顺序解耦（IV Bank 显式 IV，见硬件架构条目），使预加密可乱序执行。调度器识别两类空闲：**true idle**（两个 fault batch 之间的 driver 线程睡眠期，平均占 driver 执行时间 87%）与 **false idle**（batch 内 fault preparation 阶段——取 fault、预处理的时段，AES 指令未执行）。候选页选择策略决定变体：F-LÆGIS（false idle + fault buffer 下一批条目）、IR-LÆGIS（true idle + 随机 CPU 驻留页）、IN-LÆGIS（true idle + fault buffer 条目）、IFN-LÆGIS（false+true idle 全用：先 fault buffer 候选，剩余空闲顺序预加密 CPU 驻留页）。预加密完成的页到达时直接标记 ready、跳过关键路径加密，且批量提交提升 PCIe 突发利用率。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
预加密调度器运行在 CPU driver 侧（伴随 nvidia-uvm fault 服务链，加密走 Linux Kernel Crypto API，实测 1.3 GB/s），伪代码：
```
// 每处理完一个 fault batch i 后，调度器进入 idle 窗口
on_batch_dispatched(batch_i):
  # false idle: fault preparation 期间（CE 尚在处理 batch_i 的数据）
  for pg in next_fault_buffer_entries:          # F/IFN：取 fault buffer 下一批候选
      if pg.preencrypted == false:
          iv = ivbank_lookup(pg.id)             # CPU IV Bank 按 ID 重建 IV
          preencrypt(pg, AES_GCM(K_h2d, iv))    # 提前加密，标记 ready
  # true idle: 批次之间（driver 睡眠窗口，平均占 87%）
  while driver_idle:
      pg = IR: random_cpu_page() | IN: next_fault_buffer() | IFN: next_seq_cpu_page()
      preencrypt(pg, AES_GCM(K_h2d, ivbank_lookup(pg.id)))  # 结果直接提交
on_fault_served(pg):
  if pg.preencrypted: mark_ready(pg)            # 免关键路径加密
  else: encrypt_on_critical_path(pg)            # 未预加密页照旧
```
（Annotations：候选集 S_a 为全部 UVM 管理页；预加密页移出 S_a；DMA 时附带 MAC||ID；IV 从 IV Bank 随机访问故预加密顺序无需匹配实际访问顺序，这是与 PipeLLM 预测式加密的关键差异——LÆGIS 密文总是可提交、无误预测 NOP/丢弃。）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：调度器在 GPGPU-Sim+UVMSmart 中建模——显式建模 fault preparation 时间、driver 线程 idle 窗口（按真实硬件 profile 注入），预加密线程利用 idle 窗口执行 AES；评估四种策略 × 默认（Pt=51%）/aggressive（Pt=1%）预取。结果：F-LÆGIS 1.51×、IR-LÆGIS 1.38×、IN-LÆGIS 2.17×、IFN-LÆGIS 2.22×（最大 3.13×）；aggressive 预取下 pIFN-LÆGIS 2.74×（最大 5.05×）、driver active 占比 88.3%；与硬件加速对比（MT 多线程 -35%、X-Baseline 1.19×）证明"利用空闲窗口的机会性预加密"是核心杠杆，无需更快加密硬件即可逼近 Ideal（差距 5.8%）。使用场景：任何 fault-driven 且存在 driver 空闲的 UVM 机密迁移路径；与 UVM 预取/预测研究（TBNp、预测器）正交兼容（IR-LÆGIS 随机选页仍得 1.38× 证明不依赖预测）。

涉及论文标题：
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing
