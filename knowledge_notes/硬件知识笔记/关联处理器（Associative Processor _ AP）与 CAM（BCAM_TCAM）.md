## 关联处理器（Associative Processor / AP）与 CAM（BCAM/TCAM）

术语解释
AP 把"存储即算"组织为对内容可寻址存储器（CAM）的批量搜索+更新：任何 in-place 操作都能展开成（可能大量）位并行搜索与更新序列；搜索/更新是 AP 唯一原生操作。BAAP 把 host 侧的 AP 设计（CAPE/PUMICE）搬到 DRAM bank 旁、嵌入 bank 内 SRAM。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CAM：对全阵列并行的 key 匹配，BCAM 支持 0/1 键、TCAM 增加 don't-care（X）位；Jeloka 6T push-rule 单元中词垂直存储（column-wise）、search key 置于 wordline，匹配列 BL/BLB 保持预充高电平、失配则放电，AND 后得到 tag 位。AP 范式沿革：Ewing&Davies 1964、STARAN 1972、CAPE（HPCA 2021，web：https://par.nsf.gov/biblio/10225228-cape-content-addressable-processing-engine——7nm、32K/131K lanes、微操作约 237ps、平均 14×/最高 254× 相对面积等价 OoO CPU、RISC-V 向量扩展）；PUMICE（DAC 2023）把 AP 挂入标量流水线实现访存-计算重叠；Hyper-AP（ISCA 2020）做全栈微码优化（BAAP ASU 的微码来源）。基本操作三步（论文图 2 XOR 例）：① search 第一行=0∧第二行=1 → 置 tag；② searchacc 搜反例并把结果 OR 累积进 tag（非破坏）；③ update 把 tag 内容批量写回目标行。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
AP 链组织：多个 32x36 子阵列（32 行数据+4 行元数据）拼成传播链；位切片布局下第 i 个子阵列存所有向量元素的第 i 位，进位/中间结果沿链经反馈线传播；每列 1 个 AND 门 + tag 锁存。周期例子（论文表 IV）：ap_xor 4 周期 = 2 次 search + 1 次 searchacc + 1 次 update；ap_add 8n+2 周期位串行进位传播；ap_search 1 周期、ap_update 1 周期。BAAP 单 bank AP 存 32×96 元素向量（25% WRAM），16 芯片 DIMM 同时操作 12,288 个 32-bit 结果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CAPE 家族以 gem5 + RISC-V MinorCPU 建模；BAAP 移植到 UPMEM 底座（350MHz、VL 96–384/bank、内存后端用 UPMEM 实测时序）。使用场景：模式匹配（DirectAP）、SIMD 算术（位切片）、数据库/图/基因组查询。局限：位串行乘 4n²+4n 周期很长，必须靠 VPU 预取与 DMA 重叠补偿（BAAP 的 VPU）。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing
