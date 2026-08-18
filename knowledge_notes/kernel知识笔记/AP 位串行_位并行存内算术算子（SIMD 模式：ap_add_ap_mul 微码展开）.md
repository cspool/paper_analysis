## AP 位串行/位并行存内算术算子（SIMD 模式：ap_add/ap_mul 微码展开）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把 SIMD 算术用 CAM 搜索/更新序列实现，一次作用于子阵列列方向的全部元素（论文表 IV）：位并行 ap_and/ap_or 3 周期、ap_xor 4 周期；位串行 ap_add/ap_sub 8n+2 周期、ap_mul 4n²+4n 周期（n=位宽，按位从 LSB 到 MSB 传播进位/部分积）、ap_redsum n 周期归约、ap_eq n+4、谓词 ap_merge 4 周期。全部由 ASU 展开成微码，位切片布局下第 i 个子阵列存所有元素第 i 位。吞吐模型 P_op = VL/c_op × f（75% 配置、350MHz、VL=320：位运算 ≈37 GOPS、32-bit add ≈0.43 GOPS、mul ≈0.027 GOPS；对照 UPMEM 单 DPU roofline ≈0.35 GOPS）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SIMD 模式，位切片布局，tag 为每列 1 位的匹配结果锁存
ap_xor(vdst, vs1, vs2):                 # 4 周期
  tag  = search(vs1==0 AND vs2==1)      # 真值表第一类，1 周期
  tag |= searchacc(vs1==1 AND vs2==0)   # 反例 OR 累积进 tag，2 周期
  update(vdst, tag)                     # 按 tag 掩码位线批量写回，1 周期

ap_add(vdst, vs1, vs2):                 # 8n+2 周期，位串行进位传播
  carry = 0
  for i in 0..n-1:                      # 每位：进位扩展 2 次搜索 + 加法真值表 2 次
    s     = vs1[i] XOR vs2[i] XOR carry
    carry = majority(vs1[i], vs2[i], carry)
    update(vdst[i], s)                  # 第 i 子阵列写回
```
调度要点：长位串行乘（4n²+4n）必须与 DMA 重叠（VPU 预取），否则 bank 带宽闲置；VA（向量加）在 BAAP 上收益有限，因为 UPMEM 标量加已单周期。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与 CAPE/PUMICE 的向量 ISA 同源（web：https://par.nsf.gov/biblio/10225228-cape-content-addressable-processing-engine），BAAP 首次把它放进 DRAM 工艺约束（降频 350MHz）。使用：PrIM 的 GEMV/MLP/TS 靠位串行乘并行化 + VPU 重叠；SEL/SCAN/REDSUM 靠流式比较/归约达到带宽饱和。权衡：位串行省面积但延迟长（乘法数百~数千周期），依赖重叠与宽向量（VL 96→384 使 decode GEMV 的 VPU 收益从 39% 降至 20.9%）摊薄。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing
