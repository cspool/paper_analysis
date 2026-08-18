## 双路径误差补偿（Fine-Grained + Coarse-Grained FPMA Compensation，FG/CG 补偿）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 双路径误差补偿是 UNICORE 恢复 FPMA 近似乘法精度的机制，包含两条互补路径：(1) 粗粒度补偿（CG，Coarse-Grained）——沿用 April 等早期 FPMA 设计的思想，把下采样/近邻尾数组合的误差以 1-bit 形式注入 FPMA 结果的 mantissa 域（作为最低加法 slice 的 carry-in）；(2) 细粒度补偿（FG，Fine-Grained）——由于乘法与 FPMA 结果对给定尾数对 (M_A,M_W) 都是确定的，可在离线扩展精度域预计算残差，把残差的低位部分编码为短 bit-pattern 存入小型 LUT，运行时把 C_fg(M_A,M_W) 拼接到 FPMA 结果的 LSB 侧，等效扩展有效尾数宽度。关键发现：CG 单独在 2-bit 尾数（FP4 E1M2）粒度下无法表达修正（误差量级太小无法触发最后一位），必须由 FG 拼接恢复低位。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 计算例子（图 5）：FP8（E4M3）——精确积 66，FPMA 低估为 60；CG 1-bit 补偿把高位调到 64，FG 拼接低位 "01" 恢复 66（与全精度一致）。FP4（E1M2）——精确积 36，FPMA 输出 32；CG 无法改变结果（仍 32，误差 < 2-bit 尾数粒度），FG 拼接 "01" 后得 36。消融（Table III）：FP16 仅 CG 即近无损（11.02→UNICORE 10.98 vs FP16 10.88）；FP8 CG 不足（11.02 vs FP8 10.98），加 FG 后 10.98 与 FP8 一致；FP4 无 FG 时 PPL 崩坏到 1.1E+4–4.9E+6，FG+CG 后 11.15 与原始 FP4 完全相同。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：FG 用小型 LUT（按 (M_A,M_W) 索引的短 bit-pattern），CG 用 1-bit 进位注入（W8 融合模式利用 B 的空闲位参与粗粒度尾数加法）；FP16/FP8 等高位宽只需 CG 或 CG+FG，FP4/FP3 低比特必须 FG+CG。使用：随精度自适应——补偿级别由格式数值特征决定，使 UNICORE 在所有支持位宽（FP4/FP8/FP16）保持与对应全精度乘法一致或近一致的模型精度。开源：https://github.com/CLab-HKUST-GZ/isca53-unicore。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference
