## RPU - A Reasoning Processing Unit

- 属于编译框架的实现是什么？实验比较什么？
  提出RPU编译工具链，将PyTorch模型graph编译为RPU ISA指令流，核心设计：(1) RPU ISA：将优化过的vector-matrix和elementwise dataflows硬化为CISC-style指令，每条指令含operand addresses、tensor dimensions、data types和Pipeline Arbiter flags。硬件执行固定的streaming schedule，DMA engines deterministic push-based dataflow injection，Pipeline Arbiter flags嵌入每条指令同步memory/compute/network pipelines。Computation是push-based：pipelines在input ready时前进，不需software polling。(2) Python Compiler：轻量级compiler trace PyTorch operations（如torch.compile [1]），将torch.nn.Linear lowering为三阶段micro-kernel——Loading（config DMA for weight/KV cache transfer）、Looping（drive VMM pipeline, manage weight tile iteration）、Launching（forward activation fragments to downstream cores）。Compiler static orders所有DMA和compute指令，pre-shards和quantizes weights，generates synchronized instruction streams for memory/compute/network pipelines。(3) Deployment model：每RPU core含lightweight instruction-fetch pipeline，执行一小组long-running instructions覆盖full LLM，实现fully autonomous execution，消除GPU的host-driven offload模型。Host仅做coordination（transfer KV$ from prefill engine、处理layer间interrupt、report generated tokens/completed queries）。实验比较：论文未提供独立的compiler vs baseline对比，compiler正确性通过compiled PyTorch transformer layers在RTL model上的functional correctness和dataflow验证。

- 硬件平台是什么，配置是什么。
  RPU custom hardware。Compiler运行在host CPU（论文未指定host配置）。Compiler target为RPU Reasoning Core，生成RPU ISA指令流。RTL verification target: TSMC N16 projected to N2。

- 开源编译框架是什么。修改了什么。
  自研RPU compiler，基于Python trace PyTorch operations [1]。未基于现有开源编译框架（如MLIR/TVM/XLA），而是从零构建轻量级toolchain。核心：IR为RPU ISA（CISC-style），compile pass包括：(a) PyTorch FX graph trace → operation识别；(b) torch.nn.Linear lowering → 三阶段micro-kernel（Loading/Looping/Launching）；(c) Static DMA ordering → 按data dependency topological sort memory/compute/network instructions；(d) Weight pre-sharding → 按CU count和core hierarchy column-shard weight matrices；(e) Weight quantization → pre-quantize to MXFP4/BFP/NxFP before deployment；(f) Instruction stream generation → per-pipeline (memory/compute/network) synchronized instruction streams with Pipeline Arbiter flags embedded。Compiler static属性：不进行runtime optimization或JIT compilation。

- 开源情况。编译框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：论文未提供compiler开源。Compiler使用流程（论文Section VI描述）：
  1. Model preparation：PyTorch model (如Llama3-8B) → torch.export或FX trace导出computational graph
  2. Compilation：`python rpu_compiler.py --model llama3-8b --cu_count 64 --precision mxfp4 --seq_len 8192` → compiler执行：
     - Op recognition：识别torch.nn.Linear, attention, SiLU/GeLU, RMSNorm, RoPE等操作→映射到RPU ISA primitives
     - 以torch.nn.Linear为例：lowering为三阶段——Loading micro-kernel: 配置Memory DMA从HBM-CO读取compressed weight tile到memory buffer → Looping micro-kernel: 配置Compute DMA从memory buffer取weight tile→Stream Decoder on-the-fly dequantize→broadcast to TMAC→Compute DMA drive VMM stripe iteration → Launching micro-kernel: 配置Network DMA forward activation fragment to next core
     - DMA ordering：topological sort所有memory/compute/network DMA操作→static schedule
     - Weight sharding：按C=64 cores column-shard W ∈ R^{K×N} → 每core store W_i ∈ R^{K×N/64}
     - Weight quantization：pre-quantize to MXFP4 block format → store compressed tiles in HBM-CO
     - Instruction stream generation：每pipeline生成独立instruction stream，每条指令embed Pipeline Arbiter flags (valid counter set/check, DMA addresses, tensor dims)
  3. Deployment：compiled instruction streams load到每RPU core的instruction memory→RPU core执行long-running instruction loop→autonomous token generation (no host offload)→host仅在layer transition时接收interrupt
  4. Compiler作用：将PyTorch model graph deterministically映射到RPU ISA，static ordering消除runtime scheduling overhead，pre-sharding和pre-quantization使deployment ready。Compiler确保dataflow正确性，利用Pipeline Arbiter flags实现deadlock-free execution

