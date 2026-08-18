## MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

- 属于算法pipeline的实现是什么？实验比较什么？
  - （近似分层：本论文是 TEE 内 ORAM 的非确定性安全机制，不属于模型推理加速类算法；此处从"提出新的算法模型"角度记录）实现为 MC-ORAM 算法——面向 VM-based TEE（Intel TDX/AMD SEV-SNP）的掩码辅助+计数器非确定性方案。核心：把 TME 的确定性 AES-XTS 加密（同一物理地址同一明文产生同一密文，产生密文侧信道）转化为非确定性。每个 128 位 AES 块重排为 112 位 masked data + 16 位 counter；同一 ORAM 树节点或暂存（stash）内的所有 112 位块共享同一个 112 位随机掩码（不同物理位置由 AES-XTS 的地址 tweak 天然区分）。每次访问对 112 位数据做掩码写（data⊕mask）、16 位计数器 +1；计数器到达 2^16−1 即将溢出时执行 Refresh 算法（对整个节点/暂存重新生成随机掩码、所有计数器清零）。算法共 6 个：Algorithm 1 初始化（每节点/暂存生成 node.mask/stash.mask、计数器清零、D[i]⊕mask 写入）、Algorithm 2/3 读路径+TreeToStash（wrMask 条件写+全暂存计数器递增）、Algorithm 4/5 驱逐+StashToTree（反向，同时更新树节点与暂存计数器）、Algorithm 6 Refresh。刷新频率与访问模式无关：树节点刷新期望仅 3.05×10^−5 次/访问，暂存每 2^16/(2ZL) 次访问刷新一次（N=2^14 时每 585 次、N=2^20 时每 409 次、N=2^23 时每 356 次），摊销开销 <1% 运行时间；密文非确定性概率 1−2^−112。带宽开销仅 baseline 的 1.125×（对比 64 位交错计数器方案的 2×），存储减少 43.75%。
  - 实验比较什么：MC-ORAM vs 采用 64 位交错计数器（Obelix 风格，每 64 位数据配 64 位计数器）的 PathORAM/PathORAM+/RingORAM/RingORAM+ baseline（+ 表示采用 Oblix 的暂存优化：路径读只把目标块放入暂存，每 3 次访问额外驱逐一次防溢出），指标为平均访问延迟（1 百万次访问算术平均，含递归位置图查询+readPath+驱逐）与带宽/加速比。结果：MC-ORAM 访问延迟 0.87–40.08ms（PathORAM baseline 1.48–72.87ms），最高加速 1.82×；MC-ORAM+ 0.11–5.00ms vs 0.19–8.66ms，最高 1.77×；RingORAM 侧 MC-ORAM 0.42–19.05ms vs 0.78–33.08ms 最高 1.85×，MC-ORAM+ 0.10–3.44ms vs 0.16–5.83ms 最高 1.60×。另做：① 计数器位宽消融（4/8/16/32/64 位，16 位最优，8 位因刷新过频略慢）；② 访问模式不变性（LS 线性扫描/均匀随机/高斯/重复访问 RA 四种模式延迟几乎一致，N=2^14、B=256B）；③ 纯 masking baseline 对照（N=2^14、B=256B 时 38.4ms/访问，比 64 位计数器慢 13.5×）；④ N=2^14/2^23 × B=cacheline(512b)/256B/2048B(embedding) 全组合；⑤ SPEC CPU2017 九个 benchmark 映射（表 VI）；⑥ DLRM/Qwen-8B 安全嵌入端到端（表 VII）。
- 硬件平台是什么，配置是什么。
  - 单服务器：双路 Intel Xeon 6548Y+ CPU、512 GB DDR5 DRAM；host 与 guest 均 Ubuntu 22.04.5；guest 运行在隔离的 VM-based TEE（Intel TDX）内，TME 硬件 AES-XTS 作为唯一加密机制（ORAM 树/暂存直接放 TEE 加密内存，省去客户端额外重加密）。无 GPU/专用加速器参与。
- 模型是什么。数据集和bench分别是什么。
  - 模型/数据集：DLRM（深度学习推荐模型训练，N=2^23、B=2048B）与 Qwen-8B（LLM 推理、time-to-token，N=2^18、B=16384B）的嵌入表安全访问评估，方法学沿用 LAORAM（TEE 内 ORAM 作安全嵌入表管理器），报告"无 ORAM 执行基线延迟+ORAM 查找延迟"之和：Qwen-8B 13→25.8ms/token（1.41×）、DLRM 0.17→3.61ms/input（1.66×）。bench：SPEC CPU2017 的 9 个 benchmark（表 VI 列 povray/mcf/leela/blender/omnetpp/parest/x264/sjeng），用 Intel PIN 采集每个 benchmark 5 百万个连续数据地址轨迹，统计最大唯一缓存行数（工作集），映射到最小可容纳的 ORAM 高度 N=2^16~2^24；ORAM 自身配置：PathORAM/RingORAM Z=4、RingORAM S=3、A=4，暂存 naive 90 / 优化 10；位置图用递归 ORAM（N=2^14 单级 B=32、N=2^23 六级 B=16 每块 4 条目，根位置图 2^11 条存 TEE 内存并配 64 位计数器）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：MC-ORAM 本身未开源（截至 2026-08 联网搜索未找到公开仓库）；实现以两个开源 ORAM 参考为基础：PathORAMSimulator（https://github.com/renling/PathORAMSimulator，论文[33]）与 oram_simulator（https://github.com/wangxiao1254/oram_simulator，论文[41]）；TCB 每实现 <1000 行代码、其中 <200 行为 MC-ORAM 特有（mask/counter/refresh 元数据管理）。
  - 算法 pipeline 伪代码（以 PathORAM+MC-ORAM 一次逻辑访问 d，B=256B=每 ORAM 块 16 个 128 位 AES 块、Z=4、L=log2N 为例）：
    ```
    # 初始化 (Algorithm 1): 每节点 node: node.mask=Rand(); 计数器清零; 112 位数据初始化为 node.mask
    #   stash.mask=Rand(); 位置图 PosM[d]=Rand(); 数据 D[i] 以 D[i]⊕node.mask 写入随机块
    # 读路径 (Algorithm 2/3): 
    P = PosM[d]                              # 1) 查位置图得叶子路径
    for node in 路径P:                        # 2) 逐节点读整条路径到暂存
        for i in 1..Z:
            wrMask = [False]*|stash|
            for j in 1..|stash|: wrMask[j] = !found ∧ stash[j].empty; found ∨= wrMask[j]
            TreeToStash(stash, node[i], wrMask)   # 3) 条件写+掩码
    # TreeToStash 内每个 AES 块:
    #   if stash.ctr==2^16-1: Refresh(stash)
    #   dst = wrMask[j]·(node[i][j]⊕node.mask⊕stash.mask) + !wrMask[j]·stash[j][k].data
    #   stash[j][k]_bits = dst ∥ (ctr+1)          # 即使不写也 +1 保证密文变化
    # 4) 客户端处理数据 d; PosM[d]=Rand()          # 5) 驱逐写回 (Algorithm 4/5, StashToTree 反向, 双更新树与暂存)
    # 驱逐/读路径中若任一 16 位计数器=2^16-1:
    #   Refresh(node/stash) (Algorithm 6):
    #     new_mask=Rand(); 对每个 AES 块: dst = node[i].data ⊕ node.mask ⊕ new_mask; node[i]_bits = dst ∥ 0
    #     node.mask = new_mask
    ```
  - 张量/字节级例子（一次驱逐写回单 AES 块）：树节点旧密文前的明文表示 = (data_112 ⊕ node.mask_112) ∥ ctr_16；从暂存驱逐块到节点时 dst = (data ⊕ node.mask ⊕ stash.mask)，即把"暂存掩码域"转成"节点掩码域"，再拼 ctr+1 后由 AES-XTS(addr, ·) 加密写 DRAM；两次访问同一 (addr, data) 因 ctr 必不同（同掩码周期概率 1、跨周期掩码独立以 1−2^−112 概率不同）→ 加密前 128 位值不同 → 密文不同，消除确定性密文侧信道。
