## CUDA-Q（NVIDIA 量子编程平台与 noisy simulation 编译/运行时）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CUDA-Q（github.com/NVIDIA/cuda-quantum）是 NVIDIA 的量子编程平台：C++/Python 内核式编程模型（`__qpu__` kernel），统一管理 GPU 上的量子模拟（statevector/tensornet/density matrix 后端，底层 cuStateVec/cuTensorNet）、QPU 调用与噪声模拟，支持混合量子-经典编程与多节点 MPI 扩展。本论文用 CUDA-Q v0.11.0 作为 TUSQ 的 GPU noisy simulation baseline（后端 cuStateVec v1.12.0）。
- 在 TUSQ 中的作用：作为"朴素 noisy SVS"baseline 代表——它对每个 shot 的电路实例顺序执行完整 SVS、不消除冗余；TUSQ 平均/最大 13.38×/439.38× 快于它（198 benchmark）。TUSQ 也用它做 TNS baseline（tensornet-mps flag）。噪声模拟用法：https://nvidia.github.io/cuda-quantum/latest/examples/python/noisy_simulations.html。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- CUDA-Q noisy simulation 编译/执行流程（baseline 对比场景）：
  ```
  # CUDA-Q noisy simulation（Qiskit/CUDA-Q 皆同构）
  @cudaq.kernel
  def circuit():            # 量子 kernel 定义（编译进 CUDA-Q 平台）
      q = cudaq.qvector(30)
      ... gates ...
  noise = cudaq.NoiseModel()   # 配置 DEP/measurement 噪声通道（p=1%）
  result = cudaq.sample(circuit, noise_model=noise, shots_count=10**6)
  # 内部：每个 shot 采样噪声通道 → 固定噪声门电路 → cuStateVec SVS → 平均输出
  ```
- 对比 TUSQ：同一输入电路，TUSQ 先在 CPU 做 ECM 预采样（平均 3.97s）消除冗余（10^6 个电路实例压到 S_final ≪ 10^6），再 DFTT 树遍历复用共享前缀；CUDA-Q 直接对 10^6 个实例逐次 SVS。结果：30-qubit Adder ×10^6 shots，CUDA-Q >10 小时，TUSQ 约 820s。
- 小电路（9/13 qubit Phasecode、13-qubit QAOA p=6/8/10）非 time-critical 时 CUDA-Q 反而快（TUSQ CPU 预处理成为瓶颈，TUSQ 耗时 1.36-77.14s）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：`pip install cuda-quantum`（NVIDIA 官方平台 https://nvidia.github.io/cuda-quantum/）；噪声模型 `cudaq.NoiseModel()` 支持 depolarizing、amplitude/phase damping（Pauli twirling）、measurement error 等；后端可选 statevector/tensornet-mps/density matrix。本论文在 Perlmutter A100 上以 CUDA-Q 0.11.0 跑 198 个 Supermarq benchmark（32k-10M shots）作 baseline，并记录 >40 小时超时（∞ 柱）。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
