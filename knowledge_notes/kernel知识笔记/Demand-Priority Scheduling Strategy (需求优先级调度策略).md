## Demand-Priority Scheduling Strategy (需求优先级调度策略)

术语解释
APTMoE 提出的 CUDA kernel 调度策略，用于协调三层加载阶段（inter-stage/inter-layer/inter-expert）产生的数据移动 kernel 对同一 PCIe 带宽的竞争。由于同方向的数据移动 kernel 不能在 GPU 上并发执行，三层加载可能互相阻塞。Demand-Priority Scheduling 通过 PriorityQueue + CUDA Event 前探机制，在 kernel 启动前动态决定加载顺序。

术语是什么？
核心机制：
- **优先级分配**：inter-expert（最高，当前层实时必需）> inter-layer（中等，下层预加载）> inter-stage（最低，下个 stage 预加载）
- **Kernel 启动前调度**：由于 CUDA 不支持 kernel 中断/恢复，在 kernel launch 前决定下一个加载目标，而非在 kernel 执行中
- **CUDA Event 前探**：在 load_stream 的倒数第二个 data movement kernel 前插入 event，通过 event.query() 检测加载进度。当 event 触发时，前一个 kernel 仍在执行，隐藏 launch overhead
- **Inter-Stream 同步**：每个 model block 关联一个 event，load_stream 完成数据移动后 record，comp_stream 的对应计算 kernel 等待该 event

从kernel调度角度拆解术语。
```
# comm_scheduler.py 核心逻辑（基于论文描述）
class DemandPriorityScheduler:
    def __init__(self):
        self.queues = {
            'inter_expert': PriorityQueue(priority=HIGH),   # 当前层实时 expert
            'inter_layer':  PriorityQueue(priority=MEDIUM), # 下层预测 expert
            'inter_stage':  PriorityQueue(priority=LOW),    # 下个 stage 预取
        }
        self.comp_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()
        self.block_events = {}  # model_block_name -> torch.cuda.Event
    
    def schedule_and_launch(self):
        # 1. 选择最高优先级非空队列
        queue = self.select_highest_priority_nonempty_queue()
        
        # 2. 从队列中取预定义数量的数据移动 action
        actions = queue.pop_batch(batch_size=PREFETCH_BATCH)
        
        # 3. 在倒数第二个 action 前插入 cuda_event
        if len(actions) > 1:
            probe_event = torch.cuda.Event()
            # 先启动前 n-1 个 actions
            for action in actions[:-1]:
                self.launch_load_kernel(action)
            # 插入 probe event
            self.load_stream.record_event(probe_event)
            # 启动最后一个 action
            self.launch_load_kernel(actions[-1])
        else:
            self.launch_load_kernel(actions[0])
        
        # 4. 周期性查询 (CPU-GPU 同步)
        if probe_event.query():  # event 已触发 → 加载正在进行
            self.schedule_and_launch()  # 发起下一批加载
    
    def launch_load_kernel(self, block_name):
        # 发起 host→device cudaMemcpyAsync
        with torch.cuda.stream(self.load_stream):
            load_data(block_name)  # cudaMemcpy Host→Device
            # 记录完成 event → comp_stream 依赖
            event = torch.cuda.Event()
            event.record()
            self.block_events[block_name] = event
    
    def wait_for_data(self, block_name):
        # comp_stream 等待对应 block 的数据加载完成
        self.comp_stream.wait_event(self.block_events[block_name])
        # 执行计算
        execute_computation(block_name)
```

术语一般如何实现？如何使用？
- 基于 PyTorch 的 `torch.cuda.Stream`（双流）和 `torch.cuda.Event`（同步原语）
- PriorityQueue 用 Python `queue.PriorityQueue` 或自定义实现
- 使用 `torch.cuda.Event.query()` 的非阻塞查询 + CPU 端轮询实现 proactive scheduling
- 关键约束：同方向 cudaMemcpy 不能并发 → 必须串行化但可以提前调度
- 在 APTMoE 中位于 `comm_scheduler.py`，与 `offload.py`（加载决策）和 `R_solver.py`（分配求解）协同工作

涉及论文标题：
- APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes
