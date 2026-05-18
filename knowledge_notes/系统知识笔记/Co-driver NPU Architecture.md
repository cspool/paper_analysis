## Co-driver NPU Architecture

术语是什么？通过联网搜索让回答具体和精准。

Co-driver NPU Architecture是TZ-LLM提出的NPU驱动拆分架构，用于在ARM TrustZone TEE和REE之间安全共享移动端NPU。传统NPU driver为完整单片软件栈（Rockchip NPU driver约60K LoC，依赖Linux device/memory/interrupt/power management），完整放入TEE会严重膨胀TCB，在TEE和REE各放一份则需要driver detach-attach（~32ms切换开销）。Co-driver将NPU driver拆为：REE侧Control Plane（scheduling、power/frequency管理、统一job queue），TEE侧最小Data Plane（secure job context验证、MMIO launch、interrupt completion）。通过shadow job机制和TrustZone硬件（TZPC/TZASC/GIC）实现安全NPU time-sharing。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Co-driver NPU系统架构和secure job执行流程：

```
REE NPU Driver (Control Plane, +167 LoC shadow job):
  Unified Job Queue:
  [REE_NN_1][Shadow_1][REE_NN_2][Shadow_2]...
       ↑                    ↑
    REE app              TEE TA (via SMC)
  
  Scheduler选中Shadow_1 → SMC通知TEE
         │
         ▼ (SMC世界切换)
TEE NPU Data Plane Driver (~1K LoC, TA user-mode):
  Secure Launch Sequence:
  1. TZPC: 阻止REE访问NPU MMIO
  2. GIC:  路由NPU IRQ到TEE
  3. 等待当前non-secure NPU job完成
  4. TZASC: 允许NPU DMA访问secure memory
  5. 验证: job initialized ∧ not_issued ∧ seq_num匹配
  6. 写NPU MMIO Launch寄存器
         │
         ▼ (NPU执行secure job... NPU中断→TEE ISR)
  7. TZPC/TZASC/GIC恢复non-secure配置
  8. SMC返回REE → REE标记Shadow完成 → 继续调度
```

Shadow Job机制：TA在TEE中准备完整secure execution context（command/register序列、I/O page table、buffer地址），向REE提交"shadow job"——轻量占位job（仅job ID+metadata，无实际命令数据）。REE scheduler将shadow job与普通REE NN job统一排队，选中时SMC通知TEE，由TEE data plane driver执行secure launch。安全保障：REE可调度但不能窥探/伪造secure job；sequence number单调校验防重放/重排序；Initialized/Not-Issued状态机防任意启动。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

TZ-LLM co-driver基于RK3588 + Rockchip NPU driver v0.9.8实现。REE侧修改：NPU driver +167 LoC（shadow job scheduling）+ TZ driver +197 LoC（CMA）。TEE侧：~1K LoC NPU data plane driver（TA user-mode）+ ~112 LoC TEE OS内核修改（CMA mapping + TZASC/TZPC配置）。Total TCB仅~112 LoC（TEE OS侧），vs完整driver~60K LoC。NPU time-sharing overhead：REE NN应用额外slowdown ≤3.8%，LLM额外slowdown ≤3.0%。依赖NPU driver可分离出小data plane的前提（已在Rockchip验证，Qualcomm仅调查可行性未完整验证）。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone
