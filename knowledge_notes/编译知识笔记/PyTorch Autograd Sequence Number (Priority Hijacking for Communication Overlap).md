## PyTorch Autograd Sequence Number (Priority Hijacking for Communication Overlap)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PyTorch Autograd Sequence Number 是 PyTorch 自动求导引擎的内部机制，用于在反向传播中决定当多个 autograd 节点同时就绪（所有输入梯度已到达）时，哪个节点优先执行。默认情况下，autograd 按节点在 forward pass 中的创建顺序分配 Sequence Number——先创建（forward 先执行）的节点在 backward 中获得更高的优先级（更早执行）。在大多数场景中，这种与 forward 顺序一致的优先级是合理的。

FarSkip-Collective 发现并利用了这一机制来实现反向传播的通信-计算重叠（"Sequence Number Hijacking"）。当模型架构的连接性被修改后（FarSkip），反向传播图中的一个子块的全部 autograd 节点可以在通信输入就绪之前被处理——因为依赖被打破了。通过人工降低通向通信输入的节点的 Sequence Number（延迟其处理），同时保持子块计算节点的原始（高）优先级，autograd 引擎自然地先执行子块的反向计算，为通信争取了重叠窗口。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

PyTorch autograd 的节点优先级处理（简化）：

```
# PyTorch autograd Engine.execute() 简化逻辑:
ready_queue = PriorityQueue()  # 按 Sequence Number 排序

def execute_backward(graph):
    for leaf in graph.leaves():
        leaf.grad = 1.0
        ready_queue.push(leaf)  # Priority = node.sequence_number

    while not ready_queue.empty():
        node = ready_queue.pop()  # 最高优先级的就绪节点
        grad_outputs = [input.grad for input in node.inputs]
        grad_inputs = node.apply(grad_outputs)  # 执行 backward 计算
        for input_node in node.inputs:
            if input_node.all_grads_ready():
                ready_queue.push(input_node)
```

FarSkip 的 Sequence Number Hijacking：

```
# 默认: forward 中先创建 attention 节点，后创建 all-to-all 节点
#        → backward 中 attention grad 先于 all-to-all grad 执行
# 但这不够——attention 的输入依赖于 all-to-all 的 gradient 输出

# FarSkip Hijacking:
for node in subblock_backward_nodes:
    # 提高优先级: 让子块计算先执行
    autograd.set_sequence_number(node, PRIORITY_HIGH)

for node in comm_input_nodes:
    # 降低优先级: 延迟通向通信输入的节点
    autograd.set_sequence_number(node, PRIORITY_LOW)

# 效果: autograd 先执行子块反向计算 → 产生重叠窗口
#       在重叠窗口内异步启动通信 → 等待通信完成 → 再处理通信输入节点
```

具体执行流（FarSkip MoE 层反向传播）：

```
Default autograd order:
  Combine comm grad → Routed Expert grad → Dispatch comm grad → Attention part (a) grad

Hijacked order (FarSkip):
  Routed Expert grad → [启动 Combine comm grad async] → 
  Attention part (b) grad → ... → [同步 Combine comm grad] →
  [启动 Dispatch comm grad async] → Attention part (a) grad → ... → [同步 Dispatch comm grad]

注意: Combine backward 同步点在 Routed Expert grad 之后（紧邻依赖方），
      Dispatch backward 在 Attention grad (a) 之后（为重叠留出窗口）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **Sequence Number API**：通过 `torch.autograd.profiler.emit_nvtx` 文档中描述的 autograd 内部机制访问，非公开 API。FarSkip 通过 autograd hook 和内部 `Node.metadata` 间接修改
- **backward hook 同步**：FarSkip 在通信输入 tensor 上注册 backward hook，该 hook 在梯度被消费前调用 `handle.wait()` 同步通信
- **适用范围**：任何需要修改 autograd 执行拓扑顺序的场景，尤其适用于通信-计算重叠、异步执行的 backward pass 实现
- **替代方案**：手写整个 sub-block 的 `torch.autograd.Function.backward()`——更可靠但繁琐且易错。Sequence Number Hijacking 保留了自动 autograd 的大部分逻辑
- **局限性**：(1) 依赖 autograd 内部实现（可能随 PyTorch 版本变化）；(2) 仅影响优先级，不改变依赖——如果硬依赖存在，仍会阻塞

涉及论文标题：
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models
