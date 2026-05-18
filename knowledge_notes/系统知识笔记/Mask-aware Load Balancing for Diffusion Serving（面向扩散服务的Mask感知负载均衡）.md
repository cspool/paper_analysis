## Mask-aware Load Balancing for Diffusion Serving（面向扩散服务的Mask感知负载均衡）

术语是什么？通过联网搜索让回答具体和精准。

Mask-aware Load Balancing是FlashPS提出的面向扩散图像编辑serving的请求路由策略。传统负载均衡使用request-count或token-count来评估worker负载，但在mask-aware serving场景中不够用——因为不同请求的mask ratio差异导致computation latency与cache loading latency差异巨大。FlashPS的mask-aware scheduler使用离线拟合的线性模型，根据每个请求的mask ratio估算将其放入各候选worker后的computation latency与cache loading latency，经Bubble-free DP计算该请求在候选worker上的预计pipeline completion time，最终选择预计延迟最低的worker路由。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

FlashPS mask-aware load balancing的系统流程：

```
// Scheduler维护的per-worker状态
worker_state[i] = {
    running_batch: [req1, req2, ...],  // 当前running batch中的请求及mask ratio
    batch_size: len(running_batch),
    max_batch_size: M_i,
    cache_available: {template_id: bool}  // 该worker本地是否有template的cache
}

// 离线拟合的延迟估算模型（基于profiling）
model:
    compute_latency(mask_ratio, batch_size, model) -> ms
    cache_load_latency(template_id, worker_id) -> ms  // 考虑cache是否在本地

// 请求路由决策
function route_request(new_req, workers):
    best_worker = None
    best_latency = +inf
    
    for worker in workers:
        if worker.batch_size >= worker.max_batch_size:
            continue  // batch满则跳过
        
        // 模拟将新请求加入后worker的pipeline延迟
        simulated_batch = worker.running_batch + [new_req]
        avg_mask_ratio = mean([req.mask_ratio for req in simulated_batch])
        
        // 估算computation latency（与mask ratio成正比）
        comp_lat = model.compute_latency(avg_mask_ratio, len(simulated_batch))
        
        // 估算cache loading latency
        cache_lat = model.cache_load_latency(new_req.template_id, worker.id)
        
        // 通过Bubble-free DP计算总pipeline latency
        total_lat = bubble_free_dp(comp_lat, cache_lat, use_cache_decision)
        
        if total_lat < best_latency:
            best_latency = total_lat
            best_worker = worker
    
    return best_worker
```

路由决策的三个关键因素：
1. **Mask ratio对计算量的影响**：mask ratio越低（mask区域越小），masked-only computation越少，推测该worker的pipeline latency越低。
2. **Cache locality**：若请求的template cache已在某worker本地（host memory），路由到该worker可避免cache transfer开销。
3. **Batch slack**：worker的当前batch size需小于max_batch_size，否则无法加入新请求。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. **离线延迟模型**：在不同mask ratio和batch size组合下profiling各worker的pipeline latency，拟合线性模型。模型输入为mask ratio m和batch size B，输出estimated computation + cache loading latency。
2. **Worker状态上报**：每个worker定期（每denoising step后）通过ZeroMQ向scheduler上报当前batch组成、mask ratio分布、cache命中状态。Scheduler维护全局worker状态表。
3. **路由开销**：scheduler的mask-aware决策耗时为毫秒级，远小于denoising step时间（数百ms），不构成瓶颈。
4. **与request-count/token-count load balancing的对比**：request-count均衡不考虑mask ratio差异——将mask ratio 0.01的"小请求"和mask ratio 0.5的"大请求"视为等价，导致部分worker过载、部分空闲。token-count均衡考虑了token数但不考虑cache loading不对称。mask-aware在两者基础上进一步优化，高流量下tail latency降低最多26%。
5. **热模板缓存预分布**：对于高频templates，FlashPS可预先将cache分布到多个worker的host memory，增大路由灵活性（更多worker有cache locality）。

涉及论文标题：
- FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling
