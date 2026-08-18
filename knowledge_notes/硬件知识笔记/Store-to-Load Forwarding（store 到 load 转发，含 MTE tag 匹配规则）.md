## Store-to-Load Forwarding（store 到 load 转发，含 MTE tag 匹配规则）

术语解释
乱序处理器中年轻的 load 直接取得更老 store 的数据（不经缓存/内存）的机制；MTE 下为保持正确性，AmpereOne 要求 load 的 address tag 与 store buffer 记录的 allocation tag 匹配才允许转发，并禁止跨 tag-store 指令（写 memory tag 的指令）转发。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
一般机制：store buffer 保存未提交 store 的数据与地址；年轻 load 若其地址与某个更老 store 重叠则直接从该 store 转发数据，避免等待 store 写缓存/内存，是乱序执行消除 store-load 假相关延迟的关键路径优化。MTE 下的挑战（论文 VI-D）：tagged store 的目标 cache line 的 memory tag 尚未取回/校验（SYNC 模式），转发发生时 store 的 tag 校验结果未知；若直接转发而 store 随后 fault，被转发的数据可能来自非法访问。AmpereOne 规则：(1) store buffer 记录该 store 的 allocation tag（即 address tag）；(2) load 仅当其 address tag 与更老 store 匹配时才允许转发——此时后续 tag 校验要么两者都成功、要么 store fault（连带 squash 推测 load），正确性保持（同验同错不变量）；(3) 跨 tag-store 指令（写 memory tag 的指令）的转发被禁止，因为它会破坏该不变量。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
转发判定伪代码（Ampere MTE 语义）：
```
on_load(L):                       # 年轻 load 到达 LSU
  for S in store_buffer (oldest→newest):
    if overlap(L.addr, S.addr):
      if S.is_tag_store:  break   # 跨 tag-store 指令，禁止转发（保不变量）
      if L.address_tag == S.alloc_tag:   # tag 匹配 → 允许转发
          return forward(L, S.data)
      else:                 break # tag 不匹配 → 不能转发，等待缓存/内存
  return wait_or_cache_fetch(L)   # 无法转发则等 line+tag 校验后走缓存路径
```
正确性论证：转发时 S 的 tag 校验可能未完成；但 L 与 S 的 address tag 相同，二者指向同一逻辑 tag，故校验结果一致——S 成功则 L 也合法，S fault 则 L 作为推测指令被回滚。该规则使"许多 store 无额外延迟、其余只承受适度延迟"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：store buffer 条目扩展记录 address tag；LSU 的 forwarding 逻辑比较 load 的 address tag 与候选 store 的 alloc tag；tag-store 指令在 store buffer 中打标、被 forwarding 逻辑跳过。Ampere 首代实现存在与 MTE 交互的实现缺口：ARM MTE Performance in Practice（USENIX Sec'26）测到 SPEC 456.hmmer 最高 1.43× 开销（store-to-load forwarding 行为与 MTE 不一致），Ampere 确认并已在下一代修复——佐证此类转发规则需仔细验证。与 early line fetch（见上条）配合：store 提前取 line+tag 使多数 store 的校验早于 commit，减少"等待 tag 校验"对转发/提交的阻塞。

涉及论文标题：
- Optimized Memory Tagging on AmpereOne® Processors
