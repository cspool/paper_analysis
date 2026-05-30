## Throughput-Length Aware Request Router for KV Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

论文 "Rethinking KV Cache Compression" 提出的请求调度组件，用于混合部署环境（部分 GPU FP16 + 部分 GPU 压缩算法）中将请求路由到估计端到端延迟最小的 GPU。结合两个预测器：(1) Throughput Predictor——基于 Vidur 的 offline-profiled attention operator runtime table，查表预测每 GPU 的当前解码吞吐；(2) Length Predictor——LongFormer-based BERT classifier，预测 prompt 在目标压缩算法下的 response/prompt length ratio。路由决策 = min(prefill_time + L_response_est / T_decode)。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。

**请求路由器伪代码**：
```
function route_request(prompt):
    best = (None, inf)
    for gpu in all_gpus:
        T_prefill = throughput_table[gpu.algo]["prefill"][batch+1][prompt_len]
        T_decode = throughput_table[gpu.algo]["decode"][batch][max_kv_len]
        L_est = prompt_len * length_predictor.predict(prompt, gpu.algo)
        e2e = prompt_len / T_prefill + L_est / T_decode
        if e2e < best.e2e: best = (gpu, e2e)
    return best.gpu
```

**实验**（Table 8, 4×A6000, LLaMA-7B, ShareGPT 1000 req, Poisson RPS=10）：
- FP16 baseline avg E2E: 11.4s
- KIVI w/ Both (throughput+length router): 6.3s (1.80× speedup)
- KIVI w/ Throughput only: 7.7s (1.48× speedup)
- KIVI w/ Length only: 10.9s (0.96× — length predictor alone hurts!)
- Throughput Predictor accuracy: 85.8-88.5%, Length Predictor accuracy: 87.8-95.7%

术语一般如何实现？如何使用？

论文开源：https://github.com/LLMkvsys/rethink-kv-compression。生产部署建议：router 类似 API Gateway 层，持有 throughput lookup table（定期更新）和 length predictor model（轻量，CPU 可运行），为每个 incoming request 做 GPU 选择。关键 insight：throughput predictor 和 length predictor 必须配合使用——仅用 throughput predictor 可能路由到生成 verbose output 的 GPU，仅用 length predictor 则完全不考虑吞吐差异。

涉及论文标题：
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

---
