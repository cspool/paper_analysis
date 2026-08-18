## Mid-Circuit Measurement (MCM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Mid-Circuit Measurement（线路中途测量）是在量子线路执行过程中、尚未到最终输出前对某些 qubit 进行测量（并把结果用于后续条件操作）的操作。MCM 是非幺正（不可逆）操作：测量塌缩量子态，无法用逆门回滚。在模拟中，MCM 破坏 SVS/DFTT 的"全幺正、可逆"假设——DFTT 的反向 uncompute（乘 U†）对测量边不成立。
- 本论文中 MCM 是 DFTT+Caching 的核心动机：含 MCM 的电路若直接"关掉 DFTT"（每条叶子独立 root-to-leaf 遍历），速度退化为 naive；DFTT+Caching 在每条 non-invertible 边（一层 MCM 合并为一条边）之前的态矢量入 LIFO 缓存（容量 K），回滚跨该边时取缓存态而非求逆，恢复性能。应用场景：FTQC 逻辑级模拟（surface code 轮测量、Magic State Cultivation 验证）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- DFTT+Caching 处理 MCM 的流程（图6，两层 MCM、K=2）：
  ```
  遍历树，遇 pre-MCM 节点 → push 入 LIFO cache（容量 K）
  正向跨 MCM 边（前向 1 操作）正常计算
  反向跨 MCM 边时：不从孩子 uncompute，而是从 cache pop 该 pre-MCM 态（回滚 0 操作）
  该节点的所有孩子子树都回访完后 pop 出（不再需要）
  # K < MCM 层数时：缓存每分支"离叶子最近的 K 个 pre-MCM 节点"，对以最浅缓存节点为根的子树分别 DFTT+Caching
  ```
- 性能恢复 α(K) = (N₁ - N_{DFTT+Caching,K})/(N₁ - N₂)，N₁=DFTT 关闭的操作数、N₂=理想 DFTT 下界；surface code memory 电路（d=3/5/7、p=10^-2/10^-3/10^-4、26/64/118 物理比特，d 轮测量）：容量 3 缓存即可恢复 60%-100%；d 越大（测量轮越多）需要的缓存越大。同一层 MCM 合并成一条树边可大幅降低缓存需求。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：Qiskit/CUDA-Q 均支持 mid-circuit measurement（如 Qiskit `measure` 加条件 `c_if`/`if_test`，CUDA-Q 的 mid-circuit measurement API）；Stim 内建 surface code 电路含轮测量（rounds=R）。TUSQ 用它验证 FTQC 场景：MSC（Magic State Cultivation）d=3 的 18-qubit 电路含 MCM，原代码 1166.69s（p=10^-4）→ TUSQ 2.24s（520×）；erasure、leakage 等其他非幺正通道同样可用 DFTT+Caching 处理。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
