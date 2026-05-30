## FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - FSMoE 提出一个灵活的 MoE 分布式训练系统，通过三项核心技术优化任务调度：
    1. **MoE 模块化与统一抽象**：将 MoE 层分解为 6 个子模块（Gate、Order、I-Order、Dispatch、Combine、Expert），预实现 4 种路由函数（GShard、Sigmoid、X-MoE、Expert Choice），并通过在线 profiling 为不同 MoE 实现提供任务调度。
    2. **节点内/节点间通信与计算的协同调度**：在 MP 和 ESP group 对齐节点内 GPU 数量的常见场景下，将节点内通信（ESP-AllGather、ESP-ReduceScatter，走 NVLink/Shared Memory）与节点间通信（AlltoAll Dispatch/Combine，走 InfiniBand）以及专家计算进行流水线化（pipeline），通过 4 种 case 分类和 SLSQP 求解器确定最优流水线度（pipeline degree r）。
    3. **自适应梯度分区（Adaptive Gradient Partitioning）**：在反向传播中，将 Gradient-AllReduce 的梯度按 overlappable parts 分配到各 MoE 层，通过两阶段算法（Step 1: 贪心分配，Step 2: 差分进化优化剩余梯度分配）最大化隐藏梯度同步通信开销。
  - 实验比较 FSMoE 与 Tutel（w/ PipeMoE）、DeepSpeed-MoE 在配置 MoE 层和真实 MoE 模型（GPT-2 MoE、Mixtral-7B、Mixtral-22B）上的每迭代训练时间加速比。

- 硬件平台是什么，配置是什么。
  - **Testbed-A**：48 GPU 集群（6 节点），每节点 8×NVIDIA RTX A6000 @1.46GHz, 48GB，NVLink 112.5GB/s (4x)，Mellanox MT28908 @200Gb/s InfiniBand，PCIe 4.0 x16，CPU Dual Intel Xeon Platinum 8358 @2.60GHz，512GB DDR4。
  - **Testbed-B**：32 GPU 集群（8 节点），每节点 4×NVIDIA RTX 2080Ti @1.35GHz, 11GB，无 NVLink，Mellanox MT27800 @100Gb/s InfiniBand，PCIe 3.0 x16，CPU Dual Intel Xeon Gold 6230 @2.10GHz，512GB DDR4。
  - 软件环境：Ubuntu 20.04, CUDA 11.3, PyTorch 1.12, NCCL 2.12。
  - 并行配置：Testbed-A 上 N_MP=N_ESP=8；Testbed-B 上 N_MP=N_ESP=4。N_EP 等于节点数（6 或 8）。

- 模型是什么。数据集和bench分别是什么。
  - 真实模型：GPT-2 XL MoE、Mixtral-7B (7 layers on Testbed-B due to memory limit)、Mixtral-22B (33 layers on Testbed-A)。
  - 配置层实验：1458 个不同 MoE 配置组合，参数空间为 B∈{1,2,4}, N_heads∈{8,16,32}, L∈{512,1024,2048}/{256,512,1024}, M∈{1024,2048,4096}, N_hscale=H/M∈{2,3,4}, f∈{1.2,2.4,*}, ffn-type∈{simple,Mixtral}。
  - Benchmark：per-iteration training latency（ms），speedup vs baseline。
  - 数据集：论文使用语言模型训练的标准流程，具体数据集名称论文未明确说明（以 causal language modeling 和 masked language modeling 为训练目标）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：代码开源在 https://github.com/xpan413/FSMoE。
  - **系统架构解释**：

  FSMoE 基于 PyTorch + C/C++/CUDA 扩展实现，将 MoE 层模块化为 6 个子模块并支持自定义：

  ```python
  # 使用 FSMoE 构建 MoE 层
  from FSMoE import LinearGate, SimpleOrder, MOELayer
  gate_impl = LinearGate()
  order_impl = SimpleOrder()
  moe_module = MOELayer(gate_impl, order_impl, **kwargs)
  # moe_module 可作为普通 nn.Module 使用

  # 自定义 Expert 和 Hook
  from FSMoE import ExpertBase, CallbackBase
  class CustomizedExpert(ExpertBase):
      def do_experts(self, args): pass
  class CustomizedCallBack(CallbackBase):
      def before_moe_start_hook(self, args): pass
  ```

  **算法pipeline 执行流程**（DP+MP+EP+ESP 混合并行下的反向传播，pipeline degree r=4）：

  输入 tensor X 被切分为 r 个 chunk，在流水线中依次处理：

  ```
  # 对于每个 chunk i (0 ≤ i < r):
  for i in range(r):
      # 阶段1: 节点内 AllGather (ESP-AllGather)
      X_i_ag = AllGather(X_i)        # intra-node, NVLink

      # 阶段2: 节点间 AlltoAll Dispatch
      X_i_disp = AlltoAllDispatch(X_i_ag)  # inter-node, InfiniBand

      # 阶段3: 节点内 ReduceScatter (ESP-ReduceScatter)
      X_i_rs = ReduceScatter(X_i_disp)     # intra-node

      # 阶段4: Expert Computation
      Y_i = ExpertCompute(X_i_rs)          # GEMM on GPU

      # 阶段5: 节点内 AllGather (ESP-AllGather)
      Y_i_ag = AllGather(Y_i)              # intra-node

      # 阶段6: 节点间 AlltoAll Combine
      Y_i_comb = AlltoAllCombine(Y_i_ag)   # inter-node

      # 阶段7: 节点内 ReduceScatter (ESP-ReduceScatter)
      Y_i_rs = ReduceScatter(Y_i_comb)     # intra-node

  # Gradient-AllReduce 与最后一个 chunk 的 ESP-AllGather/ReduceScatter 及 expert 计算重叠
  GradientAllReduce(grads)
  ```

  **性能模型**（线性建模）：
  ```
  t_{a2a,r} = α_{a2a} + n_{a2a}/r · β_{a2a}
  t_{ag,r}  = α_{ag}  + n_{ag}/r  · β_{ag}
  t_{rs,r}  = α_{rs}  + n_{rs}/r  · β_{rs}
  t_{exp,r} = α_{exp} + n_{exp}/r · β_{exp}
  ```
  其中 α 为启动时间，β 为每字节/每单位计算量的传输时间，n 为通信量或计算量。

  **最优流水线度求解**（Algorithm 1: FindOptimalPipelineDegree）：
  ```
  输入: α_{a2a}, β_{a2a}, n_{a2a}, α_{ag}, β_{ag}, n_{ag},
        α_{rs}, β_{rs}, n_{rs}, α_{exp}, β_{exp}, n_{exp}, t_{gar}
  输出: r, t^{moe}

  1. r1, t1 = solve(f_1)  // Case1: 节点间通信主导，SLSQP求解
  2. r2, t2 = solve(f_2)  // Case2: 专家计算主导
  3. r3, t3 = solve(f_3)  // Case3: AlltoAll通信主导
  4. r4, t4 = solve(f_4)  // Case4: 节点内通信主导
  5. r = candidates[argmin(t1,t2,t3,t4)]
  6. return r, min(t1,t2,t3,t4)
  ```

  **自适应梯度分区两阶段算法**：
  - Step 1：以 t_{gar}=0 优化各 MoE 层流水线度，计算 overlappable parts 时间 t_{olp}，贪心分配梯度：n_first^i = g_grad^{-1}(min(t_grad(n_grad^{i-1}), t_{olp}^i))
  - Step 2：对剩余梯度，差分进化算法求解 min Σ f_moe^i(t_grad(x_g^i))，将剩余梯度最优分配到各层。

  **前向/反向分别调度**：前向 r_fwd 和反向 r_bwd 独立优化（反向计算量约为前向 2 倍，且含 Gradient-AllReduce），912/1458 配置下前反向最优度不同。
