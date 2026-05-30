## Asynchronous Collective Communication (async_op / CUDA Stream Overlap)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Asynchronous Collective Communication 是在 GPU 计算进行期间启动并执行集合通信操作的技术，通信在后台运行而不阻塞计算流。PyTorch 中 `handle = dist.all_reduce(tensor, async_op=True)` 启动通信后立即返回 handle，通信在 GPU 上异步执行，需要结果时 `handle.wait()` 同步。

关键机制：
- **CUDA Stream 分离**：通信 kernel 和计算 kernel 在不同 Stream 执行，GPU SM 调度器同时从多 Stream 取指令
- **通信仅占用部分 SM**：NCCL/RCCL 通信 kernel 使用部分 CUDA cores 做数据打包/解包，Tensor Cores 和大部分 CUDA Cores 仍可用于计算
- **原地完成**：all-reduce 通常是 in-place 的，结果直接写入原 tensor 内存

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FarSkip-Collective 训练中的异步 all-to-all（含 Sequence Number Hijacking）：

```
# 前向: 异步 Dispatch + Combine
class AsyncAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, group):
        handle = dist.all_to_all(output, input, group=group, async_op=True)
        ctx.handle = handle  # 存储供 backward 使用
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.empty_like(grad_output)
        handle = dist.all_to_all(grad_input, grad_output, group=ctx.group, async_op=True)
        ctx.backward_handle = handle
        return grad_input, None

# Sequence Number Hijacking (反向传播优先级重排):
# 子块计算节点: 高 Sequence Number → autograd 优先执行
# 通信输入节点: 低 Sequence Number → autograd 延后执行
# 效果: 在通信等待期间先执行子块计算，最大化重叠窗口
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- PyTorch: `dist.all_reduce(t, async_op=True)` → `Work` handle → `handle.wait()`
- CUDA Stream: `with torch.cuda.stream(comm_stream): handle = dist.all_reduce(t)`
- CUDA Graphs: 使用 PyNCCL 的 graph-compatible API (标准 torch.dist 不支持 graph capture)
- Overlap 前提：(1) 架构存在依赖断裂点；(2) 通信与计算无数据依赖；(3) 通信时间 ≤ 可重叠计算时间

涉及论文标题：
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models
