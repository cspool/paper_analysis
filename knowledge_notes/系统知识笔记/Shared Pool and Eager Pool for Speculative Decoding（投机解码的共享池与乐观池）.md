## Shared Pool and Eager Pool for Speculative Decoding（投机解码的共享池与乐观池）

术语是什么？通过联网搜索让回答具体和精准。
Shared Pool和Eager Pool是SADDLE提出的两种token缓冲池，用于实现prediction-verification解耦的异步speculative decoding pipeline。Shared Pool (1KB, CAM-based) 跨micro-batch聚合所有请求生成的draft tokens：各micro-batch的DLM逐生成draft token后存入Shared Pool，当总token数达GPU verification capacity C (=512)或GPU空闲时触发TLM并行验证。Eager Pool (1KB, 按micro-batch划分，每pool最多512 tokens) 实现乐观执行：TLM正在验证Shared Pool中token时，DLM基于"当前token将被接受"的乐观假设继续为H_t仍高于τ的请求生成新draft tokens，暂存到Eager Pool；验证通过后迁入Shared Pool，被拒绝则丢弃并以TLM修正token重开drafting。两Pool的token migration为lightweight on-chip memory operation，每verification iteration后刷新，所需capacity极小（~1KB），不会成为系统瓶颈。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。
Pool-based异步pipeline执行流程：
```
// 初始状态: Shared Pool = {}, Eager Pool = {}

// Phase 1: DLM Drafting → Shared Pool积累
for each micro-batch concurrently:
    Draft Generator:
        while H_t > τ:
            x_t = DLM.next_token()
            SharedPool.insert(x_t, request_id, position)
            
// Phase 2: Shared Pool满 → 触发TLM Verification
if SharedPool.token_count >= C or GPU_idle:
    verified = TLM.verify(SharedPool.all_tokens)
    // DLM同时为未停止的请求继续生成新tokens → Eager Pool
    for request with H_t > τ:
        EagerPool.insert(DLM.next_token(), request_id)
        
// Phase 3: 验证结果处理
for each request:
    if all_draft_tokens_accepted(request, verified):
        SharedPool.merge(EagerPool.pop(request))  // Eager → Shared
    else:
        EagerPool.discard(request)  // 丢弃无效tokens
        restart_drafting(corrected_token)

// Phase 4: 循环
```
Shared Pool的CAM设计支持高效token存储、索引和检索。系统调度策略为贪心：Manager优先验证最可能被接受的draft tokens（跨请求排序），最大化有效吞吐。

术语一般如何实现？如何使用？
SADDLE在Manager中实现：Shared Pool和Eager Pool各1KB CAM/SRAM，Eager Pool按micro-batch数量细分。每个micro-batch分配一个Draft Generator，内含Controller（管理H_t）和Eager Pool分区。Manager的Scheduler监控Shared Pool fill level决定verification触发时机，并协调Eager Pool的migration/discard操作。消融实验：SADDLE-p (+Shared Pool)比仅用自适应draft的SADDLE-d吞吐提升1.52×；加入Eager Pool (SADDLE-s) 再提升1.24×。

涉及论文标题：
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems
