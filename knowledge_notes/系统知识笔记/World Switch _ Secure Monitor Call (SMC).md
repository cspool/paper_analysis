## World Switch / Secure Monitor Call (SMC)

术语是什么？通过联网搜索让回答具体和精准。

World Switch（世界切换）是ARM TrustZone体系的核心机制，指处理器在Secure World和Normal World之间切换。切换由SMC（Secure Monitor Call）指令触发，处理器进入EL3（Exception Level 3，最高特权级）的Secure Monitor，执行上下文保存/恢复和NS（Non-Secure）状态位翻转。在TZ-LLM中，世界切换用于：(1)REE和TEE之间的LLM请求/响应传递；(2)co-driver NPU中secure job控制权转移（REE scheduler→TEE data plane→REE scheduler）；(3)CMA allocation/deallocation请求转发。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

TZ-LLM co-driver NPU场景的World Switch流程：

```
// 进入Secure World（REE→TEE）
REE NPU scheduler选中Shadow Job
  → REE driver: SMC #SECURE_NPU_JOB
  → CPU trap to EL3 Secure Monitor
  → 保存Normal World上下文（x0-x30, SP_ELx, ELR_EL3, SPSR_EL3）
  → NS=0（切换到Secure World）
  → 恢复Secure World上下文
  → ERET → TEE OS → TEE NPU data plane driver

// TEE driver执行secure launch sequence后返回
  → SMC #RETURN_TO_REE
  → EL3 Monitor保存Secure上下文→NS=1→恢复Normal上下文
  → ERET → REE driver → 标记Shadow Job为"已提交"

// NPU完成中断到达（已配置为Secure Group 0）
NPU IRQ → GIC路由到Secure → EL3 → TEE OS → TEE driver ISR
  → completion处理 → 恢复TZPC/TZASC/GIC → SMC #RETURN_TO_REE
```

世界切换开销构成：(1)SMC指令trap-to-EL3开销；(2)Monitor上下文保存/恢复；(3)TLB/cache影响——世界切换不自动flush TLB，但NS位变化可能导致后续访问需重建TLB映射。传统baseline中TEE和REE各放完整NPU driver需要driver detach-attach（~32ms），TZ-LLM通过TZPC/TZASC/GIC直接配置将切换降至寄存器写入的微秒级。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Secure Monitor实现：(1)ARM Trusted Firmware-A (TF-A)——开源参考实现；(2)Qualcomm QTEE Monitor——闭源；(3)OpenHarmony TEE Monitor——TZ-LLM使用。SMC调用约定遵循ARM SMC Calling Convention (DEN0028)。世界切换本身不是TZ-LLM的TTFT bottleneck（单次SMC延迟远小于参数加载和解密），关键是减少切换次数（co-driver统一调度仅secure job执行时短暂切换）和每次切换的附带开销（硬件配置替代driver detach-attach）。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone
