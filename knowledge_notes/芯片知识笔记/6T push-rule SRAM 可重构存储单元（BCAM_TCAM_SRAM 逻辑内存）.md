## 6T push-rule SRAM 可重构存储单元（BCAM/TCAM/SRAM 逻辑内存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Jeloka 等（JSSC 2016，28nm FDSOI，web：https://ieeexplore.ieee.org/document/7400984）提出用标准 6T push-rule bitcell 实现 BCAM/TCAM/SRAM 三态可重构存储，阵列面积比传统 10T/16T CAM 小 2–5×：词按列存储（column-wise）、搜索数据放 wordline（与常规 CAM 相反）；wordline 拆成 WLL/WLR 各驱动一个存取管（仅 DRC 合规金属改动、无面积惩罚）；BCAM 搜索时 WLR=key、WLL=keȳ，匹配列 BL/BLB 保持预充高电平、失配放电，两个单端灵敏放大器输出 AND 成 match 信号（即 tag 位）；支持片上在线 SRAM/BCAM/TCAM 切换。实测：64×64 BCAM 370MHz@1V、0.6 fJ/search/bit，TCAM 0.56 fJ/search/bit。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
BAAP 的用法（论文 §II-C/§III-C）：把 UPMEM 的每 bank WRAM 宏替换为该单元。SRAM 模式下行读写走常规差分信号；CAM 模式由写 CSR 把灵敏放大器从差分切到单端（同一对晶体管两种接法、纯控制线变化、无微架构状态需排空，Jeloka 测试芯片已验证在线切换且 SRAM 模式性能不受影响）；外围仅每 BL/BLB 加 1 个 AND 门 + tag 锁存，即可使子阵列作为 CAM/关联处理器。DRAM 侧只改 bank 旁 SRAM 宏，不动 DRAM subarray、灵敏放大器与外部时序接口，新增结构全部为短距离 pitch-matched 走线——这是其能在 3 层金属的 DRAM 工艺落地的前提。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
面积/能量建模（论文 §IV-A）：FN-CACTI（7nm）给普通 SRAM scratchpad 基线，Synopsys Design Compiler + ASAP7 PDK 综合 AP 链 RTL（32x36 子阵列 SPICE 校准，含 tag/累加/归约/中间结果传播外围），得出 AP 化存储相对普通 SRAM 开销 1.2281×，据此把被重配置的容量折算为向量长度（c·p/122.81，25% WRAM → VL=96）。同单元的其它使用者：Compute Caches（HPCA'17）、Neural Cache、Duality Cache、EVE；GSI Technology APU 用更保守的 12T 实现。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing
