## Buffer Table（BT，缓冲表 / 软件页表）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Buffer Table 是 MANATEE 运行时 page manager 的软件页表：条目数 = SPM page frame（颜色）数，每个条目记录对应 frame 当前驻留的 NVM 页号。其作用是在每次 load/store 前判定"编译器预测驻留的页是否真的在对应 frame 中"，从而处理投机着色的 misspeculation。
- 关键设计：编译器为每条 load/store 插桩 (页号, 颜色) hint，运行时按颜色直接索引 BT 对应条目（无需查表搜索），命中则按页内偏移访问 SPM，未命中则加密驱逐旧页、解密载入新页并更新 BT。这让每次访问的驻留判定降到"一次直接索引 + 一次比较"的开销。
从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 伪代码（page manager 处理一条 store）：
```
store(addr, val, pageNum, color):
  frame = color                       # 编译器 hint 直接定位 frame
  if BT[frame] == pageNum:            # 命中：页已驻留
      SPM[frame][offset(addr)] = val
      WTQ.enqueue(frame, pageNum)     # 记脏
  else:                               # 未命中（misspeculation）
      if BT[frame] != EMPTY:
          encrypt_and_persist(SPM[frame], BT[frame])   # 加密驱逐旧页
      page = decrypt(fetch_NVM(pageNum))               # 解密载入新页
      SPM[frame] = page; BT[frame] = pageNum
      if is_store: WTQ.enqueue(frame, pageNum)
```
- 例子（论文 Figure 8）：SPM 有 5 个 frame、BT 条目 5 个；`store m1 100` 携带 (page=3, color=1) → 检查 BT[1] 是否为页 3，是则命中直接写 SPM[1] 并记 WTQ。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：运行时数据结构（软件），与 WTQ 配合；page manager 代码 491B。BT 依赖编译器 metadata（每条指令的页号/颜色）实现无查找访问。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
