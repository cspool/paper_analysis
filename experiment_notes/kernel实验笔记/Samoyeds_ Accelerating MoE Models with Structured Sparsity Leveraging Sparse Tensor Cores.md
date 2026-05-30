## Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：基于CUDA的**双端稀疏-稀疏矩阵乘法专用kernel**，利用NVIDIA Sparse Tensor Core（SpTC）的`mma.sp` PTX指令实现稀疏计算加速。核心kernel设计：(1) **双端稀疏数据格式**：权重端采用(N,M,V)格式——M×V块内保留N个Sub-Row，每个Sub-Row内2:4 element-wise稀疏；激活端采用vector-wise稀疏通过SEL选择数组记录routing结果。(2) **3-step hierarchical tiling**：step0为thread block tile（$m_b \times n_b$），step1为warp tile（$m_w \times n_w$），step2为SpTC指令tile（$m_i \times n_i$），K维度$K_b$受V约束需较小以避免精度损失。(3) **Data stationary优化**：引入中间寄存器$C_{IR}$，每$\frac{V}{k_b}$次迭代将C寄存器按indices矩阵shuffle，避免频繁global memory读写。(4) **Packing策略**：矩阵A按SpTC spec通过ldmatrix加载；矩阵B以转置形式packing，支持行内连续访问和跳过零值行；metadata矩阵采用自定义2-bit→32-bit映射packing，对齐32-bit memory transaction。(5) **Pipeline机制**：使用cp.async非阻塞拷贝实现fetch阶段和compute阶段重叠。kernel编译为动态库（NVCC），通过pybind11注册为Python模块。
  - 实验比较：(a) kernel级：238个合成尺寸（m,k,n ∈ [256,16384]）上对比cuBLAS、Sputnik、cuSPARSELt、VENOM的TFLOPs；(b) 真实模型benchmark：6种MoE模型配置的kernel吞吐量；(c) 不同dimension（m/k/n）独立scale时的吞吐量趋势；(d) break-down分析：逐步开启weight sparsity→input sparsity→layout optimization→data stationary的加速效果；(e) 与编译器方案PIT对比MoE层speedup。

- 后端平台是什么，配置是什么。
  - 主要GPU：NVIDIA GeForce RTX 4070 Super（Ada Lovelace，含SpTC及async copy/ldmatrix支持）
  - CPU：Intel i7-12700，16G×2 DDR5
  - OS：Ubuntu 22.04LTS，CUDA 12.1
  - 可移植性验证GPU：NVIDIA 3090、4090、A100 40G
  - 理论兼容：AMD MI300（CDNA3，有sparse ALU但缺async copy和collective load/store native支持）
  - kernel实现语言：CUDA + PTX inline assembly（mma.sp instruction）

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估脚本（来自artifact appendix）：
    - `./artifacts/kernel/synthetic_scripts.sh`：运行238种合成尺寸的kernel性能测试（对应Figure 12/13）
    - `./artifacts/kernel/kernel_model_config_scripts.sh`：运行真实模型配置的kernel测试
    - `./artifacts/MoE/figure14_scripts.sh`：MoE层性能测试
    - `./artifacts/model/figure15_scripts.sh` 和 `figure16_scripts.sh`：端到端模型测试
    - `./artifacts/MoE/figure17_scripts.sh`：breakdown分析
  - 修改内容：Samoyeds kernel替代了标准MoE执行流程中的GEMM操作。在vLLM/Transformers框架中，MoE层的线性投影（gate_proj, up_proj, down_proj）由Samoyeds sparse-sparse kernel替代，同时集成input permutation消除和weighted accumulation fusion。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/guqiqi/Samoyeds.git，Docker: kevinwu2017/samoyeds:1.0.0
  - 评估原理：每个kernel调用测量GPU wall-clock时间（通过CUDA event timing），计算TFLOPs = 有效FLOP（仅非零元素计算量）/ 执行时间。合成benchmark覆盖m,k,n各维度从256到16384的238种组合。
  - Kernel输入→性能输出全过程：
    ```
    输入：编码后的权重（data + indices + metadata矩阵）+ 稀疏输入矩阵B + SEL选择数组
    
    Kernel执行流程（Algorithm 1）：
    1. Init阶段：分配shared memory（A_tile, Indices, B_tile, SEL）和register（metadata, C）
    2. 加载SEL：GMEM → SMEM
    3. Pipeline loop (compute=0 to k/k_b):
       a) 加载metadata：GMEM → Register（跳过innermost tiling，直接到寄存器）
       b) Fetch阶段（异步）：
          - cp.async: 加载Indices, A_tile, B_tile: GMEM → SMEM
          - 3-step tiling：thread block tile → warp tile → SpTC tile
          - commit group for pipeline
       c) Compute阶段：
          - wait group（同步）
          - ldmatrix: SMEM → Register（按SpTC spec排列）
          - 若compute % (V/k_h) == 0: shuffle C寄存器（data stationary）
          - mma.sp: 触发SpTC执行稀疏MMA（M=16,N=8,K=32或M=16,N=8,K=16）
       d) 两步overlap（pipeline机制）
    4. 输出transposition（layout优化）：Register → GMEM（压缩格式，仅输出非零行）
    
    性能输出：TFLOPs = (2 * m * k * n * sparsity_ratio) / elapsed_time
    ```
  - Docker使用：`docker pull kevinwu2017/samoyeds:1.0.0 && docker run -it --gpus all --name samoyeds-ae kevinwu2017/samoyeds:1.0.0`，进入容器后执行上述脚本，结果用配套Jupyter notebook绘图（figureXX_plot.ipynb）。
