## SGLang（结构化语言模型程序执行框架）

术语是什么？

SGLang是一个高效的LLM推理服务框架，由LMSYS团队开发，核心创新包括RadixAttention（基于Radix Tree的KV Cache自动前缀复用）和Structured Language Model Programming（结构化生成语言）。SGLang支持PagedAttention、Tensor Parallelism、chunked-prefill等多种优化技术，是MuxWise的基线框架（基于SGLang v0.4.10post2实现）。SGLang使用Flashinfer作为底层attention kernel库，支持融合prefill+decode attention kernel。

从系统架构角度拆解术语：

SGLang的请求处理流程：
1. **请求到达**：HTTP/API请求进入SGLang server，携带prompt、sampling参数。
2. **RadixAttention匹配**：调度器将请求的prompt与前缀树（Radix Tree）匹配，识别可复用的KV Cache前缀（跨请求共享相同前缀的KV）。
3. **KV Cache分配**：通过PagedAttention机制分配物理KV块，将匹配前缀的逻辑KV映射到物理内存页。
4. **批处理调度**：SGLang支持inflight batching，将不同阶段的新老请求合并到同一decode batch。
5. **模型执行**：通过PyTorch + Flashinfer执行Transformer层的前向计算。支持tensor parallelism分布到多GPU。
6. **Token生成与输出**：采样生成token，更新KV Cache，将新token返回客户端。

术语一般如何实现？如何使用？

SGLang通过Python实现（PyTorch生态），部署方式：`python -m sglang.launch_server --model-path <model> --tp <tp_size>`。内部使用CUDA Graph优化decode iteration launch（<0.5ms），prefill使用FlashInfer的高效kernel。KV Cache管理采用PagedAttention（vLLM兼容），支持自动前缀缓存和跨请求复用。MuxWise在SGLang 0.4.10post2基础上新增GreenContext集成、layer-wise prefill、query-based sync等模块。Bullet在SGLang v0.4.6 + PyTorch 2.6.0基础上修改约4100行Python代码，将prefill和decode拆为两个独立SGLang worker进程（MPS spatial sharing），集成libsmctrl做SM masking，使用CUDA IPC共享GPU memory、ZeroMQ传递metadata、OS shared memory存储全局状态。RoMeo集成到SGLang v0.5.5，将quantized Linear layer替换SGLang原生FP16 Linear，使用CUDA Graph捕获完整quantization→GEMM→dequantization工作流，通过多CUDA stream异步并发执行cross-precision kernel。开源地址：https://github.com/sgl-project/sglang。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization
- TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling

---
