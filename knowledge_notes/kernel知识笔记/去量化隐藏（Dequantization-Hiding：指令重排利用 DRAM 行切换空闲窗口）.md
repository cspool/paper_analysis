## 去量化隐藏（Dequantization-Hiding：指令重排利用 DRAM 行切换空闲窗口）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
去量化隐藏是 FlexQ-NDP 在 NDP 指令调度层消除低比特 FP 去量化开销的技术：把与 QGroup 绑定的高精度 dequant 指令（部分和 × 激活 scale × 权重 scale，复用 PU 的乘法器执行）从原始位置"前移"到 PU 空闲窗口（free slot）内执行。观察依据：一次 DRAM 行切换（precharge t_RP + activate t_RCD = 48 cycle，DRAMSim3 参数）期间 PU 空闲，而一次 dequant 仅 8 cycle（2·t_CCDL），放入该窗口即零额外延迟。动机实验：QGroup 增大到 128 时 dequant 恰好都落在行切换窗口、贡献 0 额外延迟；小组尺寸下仅约 10% 的 dequant 天然落入 free slot，而 W-A 量化 dequant 占总延迟最高 40%。三大约束：① 数据依赖——只能前移、不能越过后续 dequant、不能越过 scale 缓冲 refill 触发的 DRAM 读（否则所需 scale 被丢弃）；② partial-sum 缓冲容量——越过 Compute 指令要扣减其产生的 partial sum 数（extra_buf 预算）；③ 空闲窗口容量——slot 已隐藏的 dequant 填满空闲窗口即失效。只用于 weight-activation 量化（weight-only 中 dequant 输出是权重值、越过计算需缓存权重，移动范围过小）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
伪代码（论文 Alg.1，逆序贪心前移）：
```
Input : 指令列表 I；未隐藏 dequant 列表 D；partial 缓冲上限 Bmax
1  S ← {}                       # 扫描全部 free slot（行切换/DRAM 读的 PU 空闲窗口）及剩余 idle cycle
2  foreach (pos, inst) in reverse(D) do          # 逆序：后到先占，保前面指令的移动弹性
3      extra_buf ← Bmax − #PartialSum(inst)
4      pos_tmp ← pos; candidate ← None
5      while pos_tmp < |I| − 1 do
6          if pos_tmp ∈ S and S[pos_tmp] > 0: candidate ← pos_tmp
7          next ← I[pos_tmp + 1]
8          if next.type == Compute:   extra_buf −= #PartialSum(next)
9          elif next.type == ReadData: extra_buf −= 0
10         else: return                # ReadScale/Dequant/WriteBack 阻断前移
11         if extra_buf < 0: return    # 缓冲容量耗尽，不可再移
12         pos_tmp += 1
13     Move(inst, candidate); 更新 S[candidate]；窗口耗尽则标记失效
```
计算过程例子（LLaMA2-7B MVM、W4A4S8、QGroup(1,16)）：PU 处理完 DRAM row-1 的权重块、切换 row-2（48 cycle 空闲）→ 把本块末尾产生的 dequant 指令前移进该窗口执行（8 cycle，仍余 40 cycle）→ 后续多个小组的 dequant 继续合并填充同一/后续窗口 → dequant 与行切换、值读取完全重叠。效果：消融实验累计贡献 ×1.18；DRAM 行切换开销平均降约 2×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译期指令重排 pass——对单个权重矩阵迭代切片后的指令列表做一次扫描（限制在单次迭代内、控制重排开销），先扫描记录 free slot 位置与空闲 cycle 数，再逆序移动 dequant 指令。使用：作为 NDP 编译流程 code generation 阶段的后置优化，配合"scale-value 交织布局"（布局决定空闲窗口周期）与缓冲分配（决定 Bmax）一起工作；仅在 W-A 量化场景启用。通用化洞察：任何"短计算指令"都可利用 DRAM 行切换的固定空闲窗口隐藏——窗口周期由数据布局决定、指令周期由缓冲容量决定，二者解耦是重排（而非强制对齐）能生效的关键。

P3-LLM 补充视角（ISCA'26，量化算子融合 Fusion，非指令重排）：与 FlexQ-NDP 的"指令重排把 dequant 塞进 DRAM 行切换空闲窗口"不同，P3-LLM 通过 operator fusion 从源头消除运行时 dequant 的粒度——把量化缩放因子折叠进另一个操作数的量化过程：线性层（Y=X@W_q）的 dequant 缩放放在矩阵乘法之后统一执行一次；Q·K^T 把 post-RoPE key cache 的 per-channel smoothing factor（SSF）元素乘进 query（先于 FP8-E4M3 量化），从而量化 key 无需解量化即可与量化 query 在 PIM 上相乘；P·V 把 per-value-head 缩放因子 S^V 融合进 attention-score（除以 S^V_max 二级缩放防 FP8-S0E4M4 越界，P·V 结果乘回 S^V_max）。效果：NPU 只需对整层输出做一次高精度 dequant（而非每个量化张量），配合 8-bit attention-score 使 attention 全模块在低精度 PIM 上执行；架构消融显示 W4A8KV4 + TEP 后加 8-bit attention-score（实现全 attention PIM 化）再获 1.2×。

涉及论文标题：
- Bringing Near Data Processing into the Low-Bit Floating-Point Era
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
