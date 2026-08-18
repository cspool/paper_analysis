## Write Tracking Queue（WTQ，写跟踪队列）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Write Tracking Queue 是 MANATEE 运行时的轻量结构，跟踪哪些 SPM 页已被写脏（dirty）：每个条目记录 (脏页号, SPM frame 号)。它的作用是让断电前只持久化被修改过的页（按页粒度加密写回 NVM），避免把整个 SPM 都加密——这是相对 NVSRAM/Mapi-Pro"整 SPM checkpoint"的核心节能点。
- 正常执行中 WTQ 满时，最老条目被逐出，对应脏页先加密写回 NVM 再复用条目；断电（V_backup 触发）时，JIT checkpoint 按 WTQ 中所有条目把对应脏页加密持久化，再保存寄存器等程序状态。WTQ 代码 1,028B，远小于整 SPM。
从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 伪代码：
```
on_store(frame, pageNum):
  SPM[frame][offset] = val
  if WTQ 满:                         # 逐出最老脏页
      (oldFrame, oldPage) = WTQ.dequeue()
      encrypt_and_persist(SPM[oldFrame], oldPage)
  WTQ.enqueue(frame, pageNum)        # 记当前脏页
on_power_failure():                  # V_backup 触发
  for (frame, pageNum) in WTQ:       # 只持久化脏页
      encrypt_and_persist(SPM[frame], pageNum)   # 4x16B 块凑 64B 原子 flush
  checkpoint(registers, heap, stack); flip(flag)
```
- 例子：程序连续写页 3、页 5，WTQ 记录 (frame1,page3)、(frame2,page5)；断电时这两页被加密写回，未修改页无需持久化。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：运行时软件队列（1,028B），与 BT、page manager 协同；脏页以 AES-XTS 页粒度（64B = 4×16B 块）原子持久化。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
