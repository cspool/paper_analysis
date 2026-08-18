## CUDAGraph（CUDA 图捕获 / 图回放）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CUDAGraph（CUDA 10 引入）允许把一系列 GPU kernel launch、memory copy、memory allocation 预录制为一个有向无环图（DAG），后续通过单次 cudaGraphLaunch 回放整图，消除逐个 kernel launch 的 CPU-GPU 同步开销与 CUDA driver 调度开销。生命周期三阶段：Graph Construction（Stream Capture 或显式 API 构建节点+依赖）→ Instantiation（编译为可执行图 cudaGraphExec_t）→ Launch（每输入重复回放，仅替换输入 buffer 指针）。核心约束：capture 时 kernel shape/launch 配置必须固定（静态输入 shape），动态 batch 需按 bucket 预编译多张图。Tetris（ISCA'26）在 decoding 阶段用 CUDAGraph 消除 kernel launch 开销（decode 每 token 都跑同一组 kernel）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Tetris decode 的 CUDAGraph 使用（vLLM 风格）
for bucket_b in [1,2,4,8,16,32]:
    graph[b] = cudaStreamBeginCapture(stream)      # 录制 decode 一步的全部 kernel
    forward(decoding_batch(b))                     # Flash Decoding + 各算子
    cudaStreamEndCapture(stream, graph[b])
    exec[b] = cudaGraphInstantiate(graph[b])
# 运行时每 decode iteration：
cudaMemcpyAsync(input_buffer, tokens, ..., stream)  # 替换输入
cudaGraphLaunch(exec[bucket(batch_size)], stream)   # 单次提交整图
```
Annotations: 每 batch bucket 一张图（动态 batch 的近似）；输入/输出用固定 buffer 地址（kernel 读新数据只需更新 buffer 内容）；回放省去每 kernel 的 host launch 与同步。
在 Tetris 中：decode 每 token 的 kernel 序列（Flash Decoding attention、FFN、采样）录制为图逐 token 重放，与 Flash Decoding 共同压低 decode 单步延迟（对 TBT 指标关键）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：CUDA Runtime（cudaStreamBeginCapture/EndCapture + cudaGraphInstantiate + cudaGraphLaunch）、PyTorch torch.cuda.CUDAGraph、vLLM 的 CUDA Graph compilation framework（按 batch size 预编译 decode 图）。使用：decode 阶段 kernel launch 开销占比高的场景（小模型、短 kernel、高吞吐 serving）；约束——需静态 shape/固定 buffer，动态控制流需 host 侧分支或 persistent kernel（FlashInfer 用 persistent kernel 兼容 CUDAGraph）。Web 证据：CUDA Graph 官方文档与 vLLM 设计文档确认 capture/instantiate/launch 生命周期。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
