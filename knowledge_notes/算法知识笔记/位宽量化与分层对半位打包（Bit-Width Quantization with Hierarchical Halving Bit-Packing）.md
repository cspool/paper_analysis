## 位宽量化与分层对半位打包（Bit-Width Quantization with Hierarchical Halving Bit-Packing）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ENEC 提出的无损定长位打包技术，把"每组按最大值计算可变位宽 + 乘除运算"替换为"两级位宽量化 + 纯位运算的 lane folding"。核心思想：(1) 位宽量化——数据块按组长度 L 分组（组内交错 scheme），若组内最大值所需位宽 ≤ 阈值 m，整组用 m 位存储；否则整组用 n 位（n 为表示所有出现指数所需的最小位数），用 1-bit bit mask 区分两种组。这用高效 bitwise OR 替代了计算昂贵的 reduction max 和乘法/除法（Ascend AIV 整数算术指令受限）。(2) 分层对半打包（hierarchical halving bit-packing，Algorithm 2）——N 元素 a-bit 数据（N=2^k, 0<a≤8）：迭代把数据块"对半折叠"，下半元素 data[i] 与上半元素 data[i+length] 左移 width 后 OR 合并进同一 lane，width 翻倍；当有效位宽超过 8 位字节边界时触发 byte 归一化——低 8 位拆出成可存字节，溢出位收集成新子块递归处理；最后补齐使总长偶对齐（16-bit aligned），再经一次折叠拼成输出流。效果：把变长打包需要的乘法/除法/规约全换成 OR、移位，压缩吞吐较基础版 +30%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 2 伪代码（N=8, a=3 举例，先演示 lane folding）：
```
data = [v0..v7], width=3, length=8, total=0
# 迭代1: length=4, data[i] |= data[i+4] << 3, width=6  # 两元素并到一 lane，6bit
# 迭代2: length=2, data[i] |= data[i+2] << 6, width=12 # 4元素并到一 lane，12bit>8 → 字节归一化
#   temp_bytes[j] = data[j] & 0xFF  (低8bit→字节)
#   data[j] >>= 8                    (溢出位留待下一轮，width=12-8=4)
# 迭代3: 对剩余溢出位继续 folding → 输出更多字节
# 末尾: total_length 补齐偶数 → 折叠拼接成 16-bit 对齐输出流
```
Annotations：位宽量化环节先保证每元素只需 ≤n 位；打包环节用 OR+shift 做"位平面压缩"（把多个窄值并进宽 lane），一次处理 2 的幂个元素天然对齐 SIMD。解压是精确逆过程（逆 gather + OR 还原）。组内超过 m 位的元素其高 (n-m) 位单独收集到 32KB buffer，满后同样打包，解压时逆 gather 放回原位。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AscendC 向量指令（元素级 OR、移位）在 AIV 上执行；GPU 移植版用 shuffle/移位指令优化 lane folding。参数 (m, L) 由离线联合搜索确定：B_exp = 1/L + n + (m-n)·p(m)^L 最小化（p(m) 为值可用 ≤m 位表示的概率，1/L 为组 mask 均摊开销；L≥16 因 Ascend 数据搬运 32 字节对齐，论文实测 L=16 最优）。使用：作为 ENEC 压缩/解压 kernel 的核心打包原语；也适用于任何"窄整数数组定长打包"场景。特点：定长 → 解压无需变长解析，天然适合无分支 SIMD。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
