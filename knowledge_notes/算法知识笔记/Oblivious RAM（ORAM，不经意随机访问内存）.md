## Oblivious RAM（ORAM，不经意随机访问内存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ORAM 是隐藏内存访问模式的安全原语（Goldreich & Ostrovsky, J.ACM 1996）：通过每次逻辑访问都伪装成对一组"看似随机"位置的访问、并把被访问块重映射到新位置，使服务器侧观察者无法区分任意两条等长访问序列（computationally indistinguishable）。本论文采用经典两方模型：*Trusted Client*（持有全部隐私关键逻辑：位置图、stash、解密/重加密、随机重映射）与 *Untrusted Server*（只按请求在结构化存储（如二叉树）中存取整条根到叶路径的密文块）。每次访问带宽被放大（PathORAM 为 log(N) 块/次），因此传统 ORAM 部署在 WAN 下带宽负担大，且客户端必须驻留本地/可信第三方。
- TEE+ORAM 变体：把 ORAM 客户端放进 VM-based TEE（Intel TDX/AMD SEV-SNP），与服务器同机部署，WAN 流量削减到只传目标块；此时 ORAM 树/暂存直接放 TEE 的 TME 加密 DRAM（确定性 AES-XTS），省去客户端额外重加密——但确定性加密产生密文侧信道（见密文侧信道条目），破坏 ORAM 不可区分性。MC-ORAM 在 TEE 内用掩码+计数器恢复密文非确定性且不修改底层 ORAM 访问序列。
- 威胁模型（论文 II）：CPU 包/缓存/TEE 特性可信；敌手可观察 DRAM 访问模式与密文内容（stash、位置图、ORAM 树）；可做内存总线嗅探。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 一次逻辑访问（PathORAM 语义）：
P = PosM[d]                       # 1) 位置图查目标块叶子
read_path(P) -> stash             # 2) 服务器返回整条根到叶路径密文，解密入 stash
process(d)                        # 3) 客户端对目标块计算
PosM[d] = Rand()                  # 4) 分配新随机叶子并更新位置图
evict(stash, P)                   # 5) 贪婪回填驱逐写回服务器
# 关键：5 步的流量形状与 d 无关（恒为整路径+全暂存扫描），故不可区分
```
- 例子：N=2^14、L=14、Z=4、B=256B 的 PathORAM，每次逻辑访问移动 14 节点×4 块路径并全扫描暂存（90 槽）；MC-ORAM 在其中叠加 112 位掩码写+16 位计数器递增，访问延迟约 0.87ms（vs 64 位计数器 baseline 1.48ms）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：树形构造（PathORAM/RingORAM）或分层/电路构造；客户端侧位置图+stash+驱逐逻辑；服务器侧只存密文桶。开源参考：PathORAMSimulator（https://github.com/renling/PathORAMSimulator）、oram_simulator（https://github.com/wangxiao1254/oram_simulator，含 PathORAM/Circuit ORAM 等）。本论文基于这两个实现开发 TDX 内版本（每实现 <1000 行，<200 行 MC-ORAM 特有）。
- 使用场景：AES 密钥恢复防护、推荐模型嵌入表（LAORAM）、LLM 嵌入表、数据库安全存储（Menhir）等所有"访问模式敏感"负载；本论文用于 TEE 内安全嵌入表管理器（DLRM/Qwen-8B 评估）。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
