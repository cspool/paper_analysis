## GPU Scoreboard and Throttle Stalls

术语是什么？通过联网搜索让回答具体和精准。
GPU Scoreboard Stall和Throttle Stall是NVIDIA GPU warp scheduler的两种主要stall类型。Scoreboard stall发生在warp需要等待data dependency（如global memory load未完成，register file data未就绪），指令无法issue因为操作数不可用。Throttle stall发生在warp需要的执行单元（如FP32 unit、Tensor Core、LD/ST unit）正被其他warp占用，指令无法issue因为硬件资源繁忙。两者之和代表了GPU因指令级依赖和结构竞争而无法发射指令的cycles占比。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
GPU warp scheduler每cycle为每个warp尝试issue指令（A100: 每个SM 4个warp scheduler，每cycle最多issue 1 instruction/warp）：
```
Warp Scheduler per cycle:
  for each eligible warp:
    instruction = decode(I$[warp.PC])
    if instruction_needs_unavailable_operand:
      → Scoreboard Stall (data hazard)
    elif instruction_needed_unit_busy:
      → Throttle Stall (structure hazard)
    else:
      → Issue instruction to execution unit
```

**Scoreboard Stall例子**：
- Warp发出LDG (load from global memory) → 数据需~300 cycles到达register
- 后续FMUL使用该load结果 → scoreboard检查发现register未就绪 → stall
- Infera的SelectKernels通过data hazard分析（collect stall cycles + running cycles）估计IPC

**Throttle Stall例子**：
- 4个warps同时需要FP32 unit执行FMUL
- A100 SM有64个FP32 CUDA cores/SMSP (4 SMSP/SM)
- 若某SMSP上4个warps都是FP32密集 → FP32 unit饱和 → throttle stall
- Infera的SelectKernels通过structure hazard分析（instruction density by type + hardware bandwidth）估计IPC

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera在SelectKernels中分析kernel hazard（data + structure）来估计IPC，用于kernel选择优化（Eq.6: min #inst/IPC s.t. TLP≥4）。通过static analysis of SASS指令来收集：(1) data hazard信息——指令间register dependency chain和memory latency；(2) structure hazard信息——各类型指令密度（FP32/FP64/TensorCore/LD/ST等）与硬件执行单元带宽的关系。这些信息送入online-learned lightweight regression model输出IPC估计。测量工具：NVIDIA Nsight Compute可捕获GPU stall reasons breakdown，论文Figure 12展示了Infera vs baselines在不同模型上的scoreboard和throttle stall cycle比例。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
