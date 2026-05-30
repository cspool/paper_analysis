## Triton (GPU Kernel Language and Compiler)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Triton是OpenAI开发的开源GPU kernel编程语言和编译器（Tillet et al., 2019, MAPL）。提供Python-like DSL，开发者用tile-based编程模型编写高效GPU kernel，Triton编译器自动处理shared memory管理、warp调度、memory coalescing等底层优化。核心理念：将kernel开发从CUDA的per-thread心智模型提升到per-tile batch级抽象——程序员指定tile尺寸和计算逻辑，编译器生成优化后的PTX/CUDA代码。

Triton的核心编程原语：(1) `triton.jit` decorator标记kernel函数；(2) `tl.program_id(axis)`获取当前thread block的grid坐标；(3) `tl.arange(start, end)`生成tile内元素索引；(4) `tl.load/tl.store`从/向global memory加载/写入tile，自动处理boundary mask；(5) `tl.dot(a, b)`执行tensor core矩阵乘（自动映射到NVIDIA wgmma或AMD mfma指令）；(6) `triton.autotune` decorator自动探索tile size/hyperparameter最优配置。

从编译框架角度拆解术语：
Triton的编译流程：

```
Python Kernel Source (triton.jit function)
  ↓ Triton Frontend
TTIR (Triton Intermediate Representation)
  ↓ MLIR Optimizations (layout conversion, loop unrolling)
TTGIR (Triton GPU IR, thread-level)
  ↓ GPU-specific lowering
LLVM IR
  ↓ LLVM PTX backend (NVIDIA) or AMDGPU backend (AMD)
PTX/AMD ISA Assembly
  ↓ GPU Driver (cuModule/hipModule)
Executable GPU Binary
```

Triton autotuner的工作原理（论文Monarch ② kernel配置为例）：
```
@triton.autotune(
    configs=[
        triton.Config({'t_n': 32, 't_r': 32, 't_p': 32}, num_warps=4),
        triton.Config({'t_n': 64, 't_r': 64, 't_p': 64}, num_warps=8),
        triton.Config({'t_n': 128, 't_r': 128, 't_p': 64}, num_warps=8),
        ... # 20+ candidate configs
    ],
    key=['n', 'r', 'b_1']  // 当这些参数变化时重新tune
)
@triton.jit
def monarch_fused_perm_bmm_kernel(...):
    // kernel body using tl.load, tl.dot, tl.store
```

在BLR论文中，Triton的关键价值：(a) 支持自定义memory layout——Triton不强制row-major，允许BLR-specific存储格式（如V重排布后的r'-first contiguous）；(b) dot()自动利用tensor core——无需手动wgmma/wmma编程；(c) index arithmetic直接实现permutation fusion——比PyTorch的einsum/torch.compile()更灵活控制数据流；(d) autotuner针对BLR形状自动探索最优tile sizes。

术语一般如何实现？如何使用？
安装：`pip install triton`。CUDA 11.x+，支持NVIDIA Volta+(sm_70)和Ampere/Hopper，AMD CDNA3/4（ROCm Triton）。论文中使用Triton 3.2.0（Jetson）和3.4.0（A40）。典型kernel开发流程：(1) 定义kernel函数，用program_id()划分grid→tile grid；(2) 用arange()+mask构建tile索引；(3) tl.load从global memory加载tile；(4) dot()/elementwise做计算；(5) tl.store写回结果；(6) 用autotune探索最优配置。限制：(a) tile尺寸需是2的幂且≥16；(b) 不支持data-dependent control flow跨warp；(c) shared memory上限由硬件决定（A40: 128KB/SM），超出会导致compilation failure；(d) 某些高级CUDA特性（如TMA, wgmma pipeline）需手动实现。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---
