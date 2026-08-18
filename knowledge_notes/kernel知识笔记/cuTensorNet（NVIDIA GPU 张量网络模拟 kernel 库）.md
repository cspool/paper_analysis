## cuTensorNet（NVIDIA GPU 张量网络模拟 kernel 库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- cuTensorNet 是 NVIDIA cuQuantum SDK 的另一核心组件：GPU 上高性能张量网络（tensor network）模拟与收缩（contraction）库，支持 MPS/MPO 张量网络态模拟（含 bond dimension 截断）、张量网络收缩路径规划（contraction path finding）与量子线路到张量网络的转换，是 CUDA-Q tensornet-mps 后端的底层引擎。本论文用 cuTensorNet v2.9.1 作为 TUSQ 的 TNS kernel 后端。
- 在 TUSQ 中的角色：TUSQ 的 ECM+DFTT 只依赖"矩阵向量乘 + 从向量采样"，因此可直接叠加在 TNS 上——DFTT 树遍历的每条边是张量网络上的门收缩，uncompute 是对应逆收缩；输出向量供频率加权采样。TNS+TUSQ 对 40-qubit 电路（bond dimension=16）平均 248.39× 加速于未优化 TNS。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TNS+TUSQ 的 kernel 调度过程：
  ```
  // 每个待模拟电路实例（ECM 输出）在 cuTensorNet 上计算（伪代码）
  tn = 把电路转化为张量网络（MPS 表示，bond dimension ≤ D）
  for edge in dfs_order(tree):
      if 正向: state = cuTensorNetContract(tn, gate_tensor)   // 门收缩，truncate 到 D
      else:    state = cuTensorNetContract(tn, gate_dagger)   // uncompute 逆收缩
  输出向量 → 按频率加权采样
  ```
- 对比：未优化 TNS（CUDA-Q tensornet-mps）每个电路实例独立从头收缩（时间随 shots 线性增长，100/1k/10k shots 外推），QFT40/Adder40/QAOA40(p=2) 分别 1119642/628889/158407 秒（40 小时超时未完成）；TNS+TUSQ 3444/2625/805 秒完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：cuTensorNet 集成于 cuQuantum SDK（https://github.com/NVIDIA/cuQuantum，pip install cuquantum-python）；CUDA-Q 通过 `--target tensornet-mps` flag 调用做未优化 TNS baseline。TUSQ 场景：ECM 在 CPU 预采样 ER/剪枝（与 SVS 相同），DFTT 在 GPU 上调度 cuTensorNet 收缩；bond dimension=16、100k shots、α=0.01、β=100。适用性：TNS 针对 SVS 内存 O(2^n) 不可行的大/深电路，与 TUSQ 冗余消除正交。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
