## RST top-utility 序列选择（Random-access Sequence Compression with Top-utility Selection）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RST 是本论文提出的核心算法：在每 4KB 页内全局搜索 utility 最高的序列，迭代选入显式 128B 序列字典，使压缩率不降（3.4× vs Deflate 3.3×）的同时支持任意 64B 块随机解压。utility 定义为"该序列若入选字典可带来的总空间节省 / 该序列消耗的字典空间"：序列 "XY" 的字典开销 D=2·9+L（2 个 9-bit 符号 + L 位长度字段），潜在节省 S=4·2·9−(4·9+D)（4 次出现压缩为 4 个索引），utility=S/D。与 LZ 家族的区别（Table 2）：序列选择是"全局 utility 最大化"而非"局部贪心最长匹配"；字典是"显式存储"（128B/页）而非"隐式全页字典"；序列长度上限 5 符号（跨迭代可捕获长重复："VWXYZ"×5→1*×5→2* 两轮压成单个索引）；解压是"逐块独立"而非"从页首串行"。
- 从算法pipeline角度拆解术语（Algorithm 1 伪代码）：
  ```
  Input: page（4096B）
  U = COUNTSEQS2TO5(page)            // 统计所有 2~5 符号唯一序列出现次数进 utility 表
  D = {}                              // 空序列字典
  page' = page
  while HASSPACE(D) and HASPOSITIVEUTILITYSEQUENCE(U):
      s* = FINDTOPUTILITYSEQUENCE(U)        // 每长度子表取最高 count 算 utility，跨长度取最大
      dict_idx = AddToDictionary(D, s*)       // 显式存入字典
      substitution_sites[] = SUBSTITUTION(page', s*, dict_idx)   // 替换所有出现为索引
      UTILITYUPDATE(U, page', substitution_sites)   // 重算受影响序列的 count
  return (D, page')
  ```
  关键点：选中一个序列会改变所有与它重叠的未选序列的 utility（每个符号只能被一个选中序列压缩），必须做"替换步骤 + utility 更新步骤"迭代；朴素全表重算需 8×10^6×64≈5 亿次操作/页，两个优化把操作降 ~1000×：① 只更新与最新选中序列重叠/含新索引的条目；② 序列长度上限 5 符号（使长度字段 L=2 位）。即便如此每页仍需 >3×10^5 次操作，串行 >100µs，必须硬件并行。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为三个并行硬件模块（见硬件架构层条目）：substitution module（CAM 匹配+overlap filter+替换+compaction）、update generator（splice-and-cancel 防双计数）、table update module（32-bank 组相联 SRAM+sorting network+FIFO）。开源：GitHub https://github.com/HEAP-Lab-VT/rst（BSD 3-Clause Clear），Zenodo artifact https://doi.org/10.5281/zenodo.19449274（C++ 参考实现 + SystemVerilog RTL + QEMU VM 镜像）；复现压缩率用 `bash regenerate_figures.sh`（~50 分钟生成 Fig.18/19）。应用：硬件内存压缩（内存控制器集成与 CXL 内存扩展场景），128B 字典已是每次压缩块访问取数的 2/3（128B 字典+64B 块=192B）。
涉及论文标题：
- Random-Access Hardware Sequence Compression
