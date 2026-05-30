## Cross-Operation Kernel Fusion (in LLM-Generated Kernels)

术语是什么？
Cross-Operation Kernel Fusion是将多个逻辑上分离的算子融合为单个Triton kernel launch的优化技术，消除intermediate tensor materialization (HBM round-trip)和kernel launch overhead。KernelEvolve通过LLM agent的graph-based search自动发现和实现fusion——agent通过系统化搜索explore不同operator compositions和tiling configurations，profiling feedback automatic ranking fusion策略的有效性。

从kernel调度角度拆解术语：
论文中三个关键fusion案例：

**Conv1d Fusion** (5 kernels → 2 kernels on H100):
```
PyTorch Conv1d:
  nchwToNhwcKernel(输入) → nchwToNhwcKernel(权重) → sm90_xmma_fprop_implicit_gemm
  → nhwcToNchwKernel → triton_poi_fused_convolution(后处理)
  5次launch, 4次layout transform + 1次compute

KernelEvolve Triton Conv1d:
  pack_conv1d_weight_kernel → conv1d_gemm_kernel
  2次launch, 无layout transform, 直接处理native 1D layout
  2.30× speedup vs conv1d, 1.62× vs conv2d workaround
```

**Optimized FM Fusion** (2 bmm → 1 kernel on H100):
```
PyTorch (torch.compile):
  独立bmm(1) → write intermediate X^TY to HBM → 独立bmm(2) → write output
  2 loads + 2 writes + 1 intermediate HBM round-trip

KernelEvolve Fused:
  Load X,Y tiles once → compute X^TY in SRAM → compute X·(X^TY) in SRAM → write output
  1 load + 1 write, intermediate stays in SRAM
  ~2× memory traffic reduction, 2-4× speedup (N ≤ 64)
```

**PFFN Fusion** (2 kernels, 3 passes → 1 kernel, 1 pass):
```
PyTorch: extern_kernels.bmm(pass1: load→bmm→write)
        → triton_per_fused_rms_norm_add_gelu(pass2: load+bias+RMSNorm stats)
        → (pass3: load+normalize apply)
KernelEvolve: single-pass: load→bmm+bias+GELU+RMSNorm+bmm+bias+RMSNorm→write
1.2-2.6× speedup on production shapes
```

术语一般如何实现？如何使用？
Agent通过搜索自动发现fusion opportunities，无需预知哪些operators可融合。关键约束是SRAM容量——fused kernel的所有intermediate results必须fit in SRAM；超出容量时fallback到unfused baseline。Deployment使用shape-specific dispatch：generated kernel用于production shapes（保证性能），fallback到PyTorch/vendor library用于out-of-distribution inputs（防止regression）。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
