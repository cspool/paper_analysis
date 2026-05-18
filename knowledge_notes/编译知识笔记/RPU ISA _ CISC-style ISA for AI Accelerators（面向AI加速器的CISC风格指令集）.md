## RPU ISA / CISC-style ISA for AI Accelerators（面向AI加速器的CISC风格指令集）

术语是什么？通过联网搜索让回答具体和精准。

RPU ISA是RPU论文提出的一种CISC（Complex Instruction Set Computer）风格的专用指令集架构，专为低延迟LLM decode的确定性dataflow执行设计。与传统GPU的RISC-like SIMT指令或NPU的VLIW指令不同，RPU ISA将优化过的vector-matrix和elementwise dataflows直接硬化（harden）为硬件原语，每条指令暴露为单个CISC-style指令。每条指令包含operand addresses、tensor dimensions、data types和Pipeline Arbiter flags（valid count set/check values）。硬件执行固定的streaming schedule（不依赖软件动态调度），computation是push-based：DMA engines deterministically inject data into buffers，pipelines在input ready时自动前进。这种设计消除了GPU的host-driven offload model和runtime kernel launch/scheduling overhead，使RPU core可完全自主执行（autonomous execution），host仅在layer边界接收interrupt。

从编译框架角度拆解：

RPU编译流程（PyTorch model → RPU ISA instruction streams）：

```
// ===== Compilation Flow =====

// Step 1: PyTorch Model Capture
model = Llama3_8B()
graph = torch.export(model)  // FX graph capture

// Step 2: Op Recognition and Lowering
for op in graph.nodes:
    if isinstance(op, torch.nn.Linear):
        // Lower to three-stage micro-kernel
        // Stage 1 (LOADING): Config Memory DMA for weight tiles
        LOADING: MEM_DMA.load(HBM_CO_addr, mem_buf_entry, valid_count=1)
        // Stage 2 (LOOPING): Drive VMM stripe iteration
        LOOPING: for each stripe: wait_activation→TMAC.compute()→tree_sum→write_reg
        // Stage 3 (LAUNCHING): Forward activation fragment to next core
        LAUNCHING: NET_DMA.send(next_core, output_frag, valid_count=consumers)
    elif isinstance(op, attention):
        generate_attention_sequence(op)  // QK^T, softmax, s(QK)V
    elif isinstance(op, SiLU/GeLU/RMSNorm/RoPE):
        generate_hp_vops_sequence(op)  // mapped to HP-VOPs

// Step 3: Static DMA Ordering
// Topological sort all DMA ops based on data dependency graph
static_schedule = topological_sort(all_dma_ops)

// Step 4: Weight Pre-sharding and Quantization
for core_i in 0..C-1:
    W_i = W[:, core_i*(N/C) : (core_i+1)*(N/C)]  // column shard
    W_i_compressed = quantize_mxfp4(W_i)  // pre-quantize

// Step 5: Instruction Stream Generation (per pipeline)
for pipeline in [MEMORY, COMPUTE, NETWORK]:
    stream[pipeline] = []
    for op in static_schedule.filter(pipeline):
        instr = RPU_ISA.encode(
            opcode, addrs, dims, dtype,
            arb_flags={valid_count, check_valid}
        )
        stream[pipeline].append(instr)

// Step 6: Deploy
// Each core receives 3 independent instruction streams + pre-sharded weights
deploy(core_i, streams, W_i_compressed)
```

Compiler key characteristics:
- 轻量级Python实现，trace PyTorch operations（利用torch.compile graph capture）
- Static ordering消除所有runtime scheduling
- Pre-sharding和pre-quantization使weights deployment-ready
- 三条pipeline各自独立instruction stream，通过buffer entry同步
- Long-running instructions：每core仅需一小套指令覆盖full LLM

术语一般如何实现？如何使用？

RPU ISA compiler作为离线工具运行：输入PyTorch model graph和deployment config（CU count, precision, sequence length）→输出per-core instruction streams。每RPU core的lightweight instruction-fetch pipeline以long-running loop方式执行指令（不per-token re-launch），实现fully autonomous token generation。与GPU编译的关键差异：GPU compiler生成细粒度RISC-like指令（每指令~1操作），依赖GPU warp scheduler动态调度；RPU compiler生成粗粒度CISC-style指令（每指令封装~千个MAC操作），依赖硬件的固定streaming schedule和Pipeline Arbiter同步。论文未开源compiler。

涉及论文标题：
- RPU - A Reasoning Processing Unit
