## Shadow Job / Secure NPU Job Scheduling

术语是什么？通过联网搜索让回答具体和精准。

Shadow Job是TZ-LLM co-driver NPU架构中的核心调度机制，用于在TEE和REE之间安全调度NPU job。Shadow Job是TEE TA在REE NPU调度器队列中放置的轻量占位job——仅包含job ID和元数据（优先级、estimated duration），不含实际NPU command/register序列。当REE scheduler选中此shadow job时，通过SMC通知TEE data plane driver，由TEE driver执行实际secure NPU job launch。Shadow Job使REE统一管理REE NN job和TEE secure job的队列调度，同时REE无法窥探或篡改secure job内容。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Shadow Job调度流程伪代码：

```
// === TEE侧: 提交Secure NPU Job ===
def submit_secure_npu_job(cmd_buf, input_buf, output_buf):
    secure_ctx = SecureExecutionContext(
        cmd_seq = cmd_buf,
        io_pt = build_iopt(),
        seq_num = next_seq_num,      // monotonic递增
        state = INITIALIZED          // 已初始化,未发行
    )
    shadow = ShadowJob(job_id, priority)
    smc_call(SUBMIT_SHADOW_JOB, shadow)

// === REE侧: NPU Scheduler ===
class NPUScheduler:
    job_queue = UnifiedQueue()  // REE NN + Shadow jobs统一队列
    
    def schedule_loop():
        while True:
            job = job_queue.pop_next()
            if isinstance(job, ShadowJob):
                smc_call(SECURE_NPU_LAUNCH, job.job_id)
                job.mark_submitted()
            else:
                npu_mmio_write(LAUNCH_REG, job.cmd_buf)
                wait_npu_completion()

// === TEE侧: Secure Launch (被SMC触发) ===
def handle_secure_npu_launch(job_id):
    ctx = secure_ctxs[job_id]
    // 安全校验
    assert ctx.state == INITIALIZED and not ctx.issued
    assert ctx.seq_num == expected_seq  // 防重放/重排序
    
    // 硬件配置
    tzpc_set(NPU, SECURE_ONLY)     // 阻止REE访问NPU MMIO
    gic_set_irq(NPU_IRQ, SECURE)   // 中断路由到TEE
    wait_current_nonsecure_job()
    tzasc_allow_npu(SECURE_MEM)    // NPU可DMA访问secure memory
    
    // Launch
    ctx.issued = True
    npu_mmio_write(LAUNCH_REG, ctx.cmd_seq)
    smc_return()  // 不等完成——中断异步通知

// TEE ISR: completion处理 → 恢复TZPC/TZASC/GIC → SMC通知REE
```

安全保证机制：
1. **防窥探**：Shadow job不含实际命令数据，REE仅见job ID/priority
2. **防重放**：monotonic sequence number——TEE验证seq_num == expected_seq
3. **防重排序**：状态机INITIALIZED→ISSUED→COMPLETED，ISSUED后不能再次launch
4. **防任意启动**：未INITIALIZED的job无法进入launch（state校验）
5. **硬件隔离**：TZPC+TZASC+GIC确保secure job执行期间REE无NPU访问权

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

TZ-LLM在RK3588实现：REE NPU driver +167 LoC（shadow job scheduling + UnifiedQueue管理 + SMC转发）；TEE侧~1K LoC NPU data plane driver（secure launch + completion handling）。Shadow job overhead：每job额外SMC往返，但TZPC/TZASC/GIC配置微秒级（vs driver detach-attach ~32ms）。NPU time-sharing对REE NN应用slowdown <=3.8%。开源：Zenodo artifact DOI 10.5281/zenodo.17213486。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

