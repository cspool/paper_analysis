## BLIMP 数据库算子内核（select/semijoin/join/aggregate kernel 与异构编排）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BLIMP 数据库算子内核是运行在 DDR bank 内 200MHz RISC-V（-V）核上的 OLAP 算子内核，与 host CPU 形成异构编排：select（FILTERTOBITVECTOR，位图输出）、hash semijoin/join（SEMIJOINPROBE，探测侧）、aggregate（非分组/分组）。内核以 1KB row buffer 粒度访存，执行 fetch-read-apply-store 流程：读入一行元素 → 逐元素应用谓词/哈希/聚合（BLIMP-S 串行、BLIMP-V 用 32×64b vALU 向量化）→ 结果 coalesce 打包进位图或输出区。host 负责建哈希表、切分 PIMDT chunk（适配 32MB bank）、relayout 载入与结果编排；跨 bank 无直连，数据交换走 host 读-写中转。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
select 内核（Algorithm 1 FILTERTOBITVECTOR）伪代码：
```
procedure FILTERTOBITVECTOR(in *d, out *b, func p)   # d=PIMDT 分区, b=位图, p=谓词, w=元素宽
    v2 ← [0,...]; eproc ← 0
    for each data row r in d:                        # 逐 1KB row buffer 行
        v1 ← FetchMem(r)                             # 读入整行（如 1024 个 8-bit 或 256 个 32-bit）
        v1 ← apply(p, v1)                            # BLIMP-S 串行 / BLIMP-V 向量化逐元素谓词
        v1 ← coalesce(v1, w, mod(eproc, 8192))       # 布尔结果打包进位图行位置
        v2 ← v1 ∨ v2                                 # 累积位图
        eproc ← eproc + ElementsPerRow(w)
        if mod(eproc, 8192) = 0 then                 # 每 8192 元素对应一个位图行 HitmapRow
            StoreMem(v2, b[HitmapRow(eproc)]); v2 ← [0,...]
    if mod(eproc, 8192) ≠ 0 then StoreMem(v2, b[HitmapRow(eproc)]); ZeroMaskRemainder(...)
```
semijoin 内核（SEMIJOINPROBE）：对每行元素 v1[i]，先向量化 hash 得桶索引 BucketRow(hash(v1[i]))，再串行探测——FetchMem 该 row buffer 桶、检查 key 是否在桶 slot 列表、命中则置位、否则沿 BucketNext 取链上下一桶（新 row buffer 读）直到 IsNull 或 hit。join 大哈希表时按分区多轮 build-probe（每轮 offload 一个分区、结果 OR 到下一轮）。调度约束：BLIMP 一次只能执行一个算子（不像 host 可并行多 kernel），compute mode 下 bank 数据对 host 锁定，下游/并行相邻操作必须 stall（host 可趁机做无关工作）。性能数据（10 亿值列）：select 位图 2.0×/12.9×（BLIMP-S/-V），值输出 2.0×/4.2×；semijoin 1.4×/2.1×、join 2.1×/3.0×；SUM 非分组 2.1×/33.7×、分组 1.9×/2.1×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现与评估：论文用 validated cycle-level simulators（prior work [25] + riscvovpsim/Imperas + DRAMSim2 DDR4 时序）模拟单 bank 内核执行（假设对称计算、取最慢 bank 周期），host 侧等价 kernel 在真实 2× Xeon Silver 4114（40 线程 AVX512 手调 C++）上计时；算子级评估只测 PIM 域计算（不含 relayout/build/加载/post-processing），端到端 SSB SF100 评估叠加全部开销。使用：查询执行器对 PIMDT 列上的算子识别是否支持，支持则走"预处理（切分/建表）→ relayout 载入 → 执行 → 部分物化/原位保留 → host 取回"工作流，不支持则回退 host；host 哈希表 build 时间与 CPU 侧 Swiss Table 相当（论文引 abseil）。内核与仿真脚本开源情况论文未明确说明，可依托同组 dovedevic/blimp（https://github.com/dovedevic/blimp）框架。

涉及论文标题：
- Taking Analytic Databases to the Bank
