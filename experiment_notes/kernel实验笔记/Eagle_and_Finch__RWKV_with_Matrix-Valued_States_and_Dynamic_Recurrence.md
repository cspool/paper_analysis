## Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Finch（RWKV-6）WKV计算的custom CUDA kernel，用于训练时加速。核心设计：不使用time-parallel的associative scan方法（虽高度并行但涉及反复HBM↔SRAM transfer），而是沿非时间维度并行，将state操作保持在fast SRAM中以减少memory transfer开销。具体而言，Finch的WKV计算可通过式(19)沿序列做并行prefix-sum，也可按式(21)-(22)以recurrent方式计算——论文选择后者，搭配SRAM-resident state management的CUDA kernel。
  实验对比Finch kernel vs Mamba kernel (2× pass, 模拟同层数) vs Flash Attention v2 (PyTorch实现)：(a) Memory Usage vs Sequence Length (A100 80GB, batch=8, D=4096, head=64, Mamba D=8192/state=16) — 图6；(b) Time vs Sequence Length — 图7。

- 后端平台是什么，配置是什么。
  NVIDIA A100 80GB GPU。Benchmark配置：batch size=8, model dimension=4096, head size=64 (Finch/Flash Attention), state dimension=16/model dim=8192 (Mamba, expansion factor=2)。Finch kernel不做time维度并行，选择沿非时间维度并行+SRAM state residency。

- 评估性能的软件/脚本是什么。修改了什么。
  自研Finch CUDA kernel。对比baseline：Mamba kernel（2× pass模拟与Transformer同层数），Flash Attention v2 (PyTorch实现)。修改：为Finch的新WKV计算（data-dependent time-varying decay w_t, matrix-valued state）编写custom CUDA kernel，核心优化是避免HBM↔SRAM反复传输，state操作保持在SRAM。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  训练代码开源：https://github.com/RWKV/RWKV-LM。

  评估原理：对预训练forward pass的实际wall-clock time和peak GPU memory进行测量：
  ```
  配置: batch_size=8, model_dim=4096, head_size=64
        序列长度从 256 扫至 16384

  Finch kernel:
    输入: x_t ∈ R^{B×H×D/H}, s_{t-1} ∈ R^{B×H×D/H×D/H}
    计算: 
      - Token shift (ddlerp): 在SRAM中计算, A∈R^{D×32}, B∈R^{32×D}
      - WKV: k_t^T · v_t → 沿非时间维度并行, s_t = diag(w_t)·s_{t-1} + k_t^T·v_t
      - state操作保持在SRAM，不反复写入HBM
      - 输出: o_t ∈ R^{B×H×D/H}, s_t ∈ R^{B×H×D/H×D/H}

  关键结果:
    - 训练时间: Finch O(N)线性扩展（与Mamba类似），16k序列时比Flash Attention快约4.2×
    - 内存: Finch比Flash Attention省约40%，比Mamba省约17%
    - 序列长度<4k时Flash Attention更快，>4k后Finch领先（因为Flash Attention内存压力更大）
  ```
