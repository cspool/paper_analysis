## Mid-Execution Authentication（执行中认证：Galois-Field 乘 + XOR 累加的镜像认证路径）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 执行中认证是 IroKnight 在加密之外提供的完整性校验：在推理执行过程中，对每次写入片上存储的数据即时计算并校验认证 hash，检测恶意篡改（加密数据被改会造成静默错误）。其数学基础来自 AES-GCM 的认证标签计算（GHASH）：每个 128-bit 数据块与认证 key 做 Galois-Field（GF(2^128)）乘法、乘积与运行中部分 hash 做 XOR 累加，最后一块累加后加密得到最终 hash。关键洞察：这个 ⟨GF-multiply, XOR-accumulate⟩ 结构与 GEMM 的 ⟨multiply, accumulate⟩（MACC）流同构——只是乘法换成二元多项式乘法、加法换成 XOR。GF 乘法中位即二元多项式系数（无进位传播）、累加只是一行 XOR（无需加法器树），因此镜像路径比 MACC 本身更简单，可与 MACC 并行计算而几乎零运行时开销。hash 随 tile/vector 存储，加载时重算并与期望值比对，不符则报警。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - 脉动阵列中的认证编排（Fig.5）：(a) 权重——写 WBUF 的同一周期，权值同时喂给 PE 内 ⟨GF-multiply, XOR-addition⟩ 对计算部分 hash，权重加载完成时全 hash 也算好并验证；write-enable 信号硬连线触发认证，保证 WBUF 任何修改都被认证；(b) 输入激活——在首列 PE 认证：加密输入激活流入首列时经 ⟨GF-multiply, XOR⟩ 算 hash，部分 hash 随波前逐 PE 下传（第一列各 PE 用自己的输入激活续算），因同一输入激活被后续所有列 PE 复用、只在首列认证避免冗余；(c) 输出激活——在末行 PE 认证：加密输出激活流出末行时从左到右算 hash 写入 OBUF；因首列与末行重叠，只有左下角 PE 需要两组 ⟨GF-multiply, XOR⟩。向量引擎（Fig.4）：每个写入子 bank 的字都过 ⟨GF-multiply, XOR⟩ 累加进 hash（如 VADD 中 A、B 载入与 C 写回各自累积 hash），写完成即得到 C 的 hash 随数据存储/传递。每个数据块附 hash 的尺寸比按 prior 安全 NPU 工作（MGX [49]）。GF 乘法分解：TEE 常用 128×128 GF 乘法器，但 IroKnight 需匹配 PE 内激活/权重的位宽，把 128×128 拆成多个 128×8 小乘法器（无进位故跨列拆分不改变阵列数据流组织）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：SystemVerilog 在每 PE 集成 Galois-Field 乘法器 + 一行 XOR（认证路径，Fig.3c），综合确认关键路径未转移到新增 GF 乘法器（仍为 WBUF 读路径）；面积/TDP 在加密基础上再增（认证变体 1.95× 面积/2.18× TDP，相对加密 1.79×/1.76×）。使用效果：认证开销只来自 hash 的片外读写流量（hash 与 MACC 并行），LLM 端到端运行时从 0.2%（仅加密）升到 3.3%（加密+认证）、能量从 ≤14% 升到 ≤18%；对 NPU 软件栈被攻破后"改密文"的攻击提供检测能力（攻击者无认证 key 无法伪造匹配 hash）。对比：TEE 靠完整性树在片外接口校验、不覆盖片上执行中的篡改；FHE 本身不防篡改。

涉及论文标题：
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving
