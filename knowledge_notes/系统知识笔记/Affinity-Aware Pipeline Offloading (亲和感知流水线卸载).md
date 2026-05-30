## Affinity-Aware Pipeline Offloading (亲和感知流水线卸载)

术语解释
Affinity-Aware Pipeline Offloading 是 APTMoE 提出的在带宽受限 GPU 节点上微调 MoE 模型的系统技术。它在传统流水线并行 + offloading（如 Mobius）基础上，引入基于 expert 热度和计算亲和性的 CPU-GPU 异构协同计算，通过将低热度 expert 的计算分配到 CPU 就地执行（而非全部加载到 GPU），减少 PCIe 数据搬移量并提升计算效率。

术语是什么？
APTMoE 的流水线并行采用与 Mobius 相同的 stage 分配策略：模型被划分为多于 GPU 数量的 stage，相邻 stage 映射到不同 GPU。但 APTMoE 的 offloading 与 Mobius 的关键区别在于：
- Mobius：每个 stage 的全部参数（包括所有 expert）在需要时从 host memory 完整加载到 GPU
- APTMoE：仅加载高热度 expert 到 GPU，低热度 expert 留在 host memory 由 CPU 计算

这使得每个 iteration 的 PCIe 数据搬移量显著减少（仅传输高热度 expert 参数），同时利用了通常闲置的 CPU 计算资源。

从系统架构角度拆解术语。
APTMoE 系统由 Static 和 Runtime 两部分组成：

**Static 部分（离线）**：
1. Profiler 在 CPU 和 GPU 上分别执行单层 MoE layer 的 fine-tuning
2. 记录每个 expert 在不同 token 数量下的执行时间（GPU time, CPU time）和数据移动时间
3. 记录每层的内存占用（memory footprint）用于 stage-to-layer mapping
4. 生成 execution time lookup table 和 layer-to-stage mapping

**Runtime 部分（在线）**：
```
# 每个 pipeline stage 的执行流程
class APTMoE_PipelineStage:
    def __init__(self):
        self.comp_stream = torch.cuda.Stream()           # 计算流
        self.load_stream = torch.cuda.Stream()           # 加载流
        self.interstage_queue = PriorityQueue(low)       # inter-stage 队列
        self.interlayer_queue = PriorityQueue(medium)    # inter-layer 队列
        self.interexpert_queue = PriorityQueue(high)     # inter-expert 队列
    
    def forward(self, micro_batches):
        # Phase 1: Inter-stage loading (stage 切换时)
        for mb in micro_batches:
            # 加载 MHA + Gate（必须，处理所有 token）
            schedule_load(MHA_blocks + Gate_ops, self.interstage_queue)
            # 加载历史高热度 expert
            for expert in historical_top_experts:
                schedule_load(expert, self.interstage_queue)
        
        for layer in self.layers:
            # Phase 2: Inter-layer loading (当前层计算时预加载下一层)
            pred_probs = predictor[layer+1](hidden_states)
            for expert in sort_by_pred(pred_probs):
                if affinity_check(expert) == GPU:
                    schedule_load(expert, self.interlayer_queue)
                else:
                    cpu_execute_queue.append(expert)  # 留在 CPU
            
            # Phase 3: Inter-expert loading (gate 执行后)
            real_probs = gate(hidden_states)
            for expert in topk(real_probs):
                if expert.affinity == GPU and expert not in GPU_memory:
                    schedule_load(expert, self.interexpert_queue)
            
            # 同步：comp_stream 等待 load_stream 完成对应 block 的数据移动
            synchronize_and_execute()
```

术语一般如何实现？如何使用？
- 基于 PyTorch 的自定义 pipeline 框架（APTMoE/Runtime/PipelineRuntime/pipeline_runtime.py）
- 使用 `torch.cuda.Stream` 管理双流重叠
- 使用 `torch.cuda.Event` 管理 inter-stream dependency
- 使用 `psutil.Process().cpu_affinity()` 绑定 CPU 核心
- 支持三种设备拓扑：C1+G4 (7核/进程), C1+G2 (14核/进程), C1+G1 (28核/进程)
- 执行命令：`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 ./main.py --pipeline=APTMoE`

涉及论文标题：
- APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes
