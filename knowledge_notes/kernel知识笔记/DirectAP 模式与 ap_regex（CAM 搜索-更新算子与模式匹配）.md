## DirectAP 模式与 ap_regex（CAM 搜索-更新算子与模式匹配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DirectAP 模式把 AP 的 CAM 语义直接暴露给程序员/编译器（论文表 IV）：ap_search(field_mask, imm) 1 周期按键置 tag、ap_searchacc 2 周期非破坏 OR 累积、ap_update(field_mask, imm) 1 周期按 tag 掩码位线写回、ap_set_tags 1 周期；field_mask 表达 don't-care（X）；伪指令 ap_regex(rdst, imm) 对长度 m 文本穷举长度 k 模式（<m−k 周期、支持通配符、模式超出单子阵列时溢出到下一子阵列、损失子阵列级并行）。数据用列连续布局，整词 CAM 匹配 1 周期完成（位切片 SIMD 需逐位 n 周期）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文算法 1（DirectAP BFS，边按列装载、FROM/TO/PROC/CURR 各占一段行）：
```
AP-BFS(ap, start_node):
  traversal_order = [start_node]
  ap.search(TO, start_node); ap.searchacc(FROM, start_node)
  ap.update(CURR, true)                    # 初始前沿
  repeat:
    ap.search({CURR,PROC},{true,false})    # 当前前沿未处理边，1 周期
    if ap.tag_popcount() == 0: break       # 前沿空，遍历结束
    ap.update(PROC, true)                  # 批量标记已处理
    tagged = ap.read_tags(); ap.set_tags(0)
    for e in tagged:
      if e.TO not in visited:
        ap.searchacc({FROM,PROC},{e.TO,false})   # 累积新邻居边
    ap.write_tags_to(CURR, true)           # 单次批量更新物化新前沿
    for n in new_nodes: ap.search(TO,n); ap.update(PROC,true)
```
关键思想：tag 位紧凑编码当前前沿，一次批量 update 物化下一前沿——把 frontier 型图遍历变成关联查询。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
适用形态（论文 §IV-E 结论）：计算可表述为对大而稀疏状态的重复 membership/邻域查询——文本模式匹配、直方图计数、图遍历。基因组案例：k-mer 计数用 bank 内穷举 CAM 匹配替代扫 DRAM+哈希表更新，对 UPMEM 2–38×；de Bruijn 图遍历 1.1–2.8×，直到跨 DPU 前沿交换经 host 中转成为瓶颈；k>21 时搜索空间 2^k 爆炸、回退多核 host。HST（直方图）类 kernel 同样受益。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing
