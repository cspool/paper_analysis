## MagicPIG: LSH Sampling for Efficient LLM Generation

- 属于Serving调度的实现是什么？实验比较什么？
  提出GPU-CPU异构系统设计，将LLM解码分为三部分：(1) GPU执行所有线性投影(MLP, W_Q, W_K, W_V, W_O)和LSH随机投影哈希码计算；(2) CPU存储LSH哈希表并执行采样检索和稀疏注意力计算(o=Softmax(qK^T/√d)V)；(3) GPU上保留sink tokens和local tokens的KV cache（on-device cache），不经过LSH采样。系统通过recursive attention技术合并GPU和CPU的注意力输出。实验比较了不同硬件配置下的吞吐量和延迟：A100 (1.5× throughput提升)、L20 (5.0× throughput提升)、RTX 4090 (3.3× throughput提升，96K context单请求54ms解码延迟)，MagicPIG可以容纳比GPU全注意力baseline大12×以上的batch size。

- 硬件平台是什么，配置是什么。
  GPU: NVIDIA A100-80GB, L20-48GB, 模拟RTX 4090-24GB (L20限制显存)。CPU: Intel Platinum 8480+ (搭配A100), Intel 8563C (搭配L20)。CPU DRAM带宽100-200GB/s，约为GPU VRAM带宽的10-20%。

- 开源Serving框架是什么。修改了什么。
  论文未使用现成Serving框架(vLLM等)，而是自建PyTorch + FBGEMM系统。GPU部分使用原生PyTorch实现，CPU注意力计算使用FBGEMM (bfloat16精度)。修改/创新点：(1) 将KV cache完整offload到CPU DRAM，通过5-10×稀疏性弥补CPU带宽劣势；(2) 在GPU上新增LSH随机投影模块(内存开销400KB~825KB)；(3) CPU上新增L张哈希表存储和查询逻辑；(4) 引入on-device cache将sink tokens和local tokens的KV保留在GPU，避免全走CPU路径。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源链接: https://github.com/Infini-AI-Lab/MagicPIG。Serving框架全流程：
  输入：用户prompt → Tokenization → embedding
  GPU端执行顺序：
    1. 线性投影：计算q = W_Q·x, k = W_K·x, v = W_V·x，以及MLP层
    2. 随机投影：q_code = Sign(q @ W)，W∈R^{d×(K×L)}，产生K×L bit哈希码
    3. 传输q_code和新生成的k,v到CPU（通过PCIe）
  CPU端执行：
    4. 查询L张哈希表：S = Query(HT, q_code)，收集碰撞的key索引
    5. 稀疏注意力计算（FBGEMM bfloat16）：计算q·K_S^T → softmax → weighted sum of V_S
    6. 结果传回GPU
  GPU端收尾：
    7. Recursive attention合并：将CPU返回的ō_cpu与GPU上的ō_gpu(on-device sink+local tokens)合并
    8. 输出投影W_O → 下一个token
  作用：突破GPU显存限制，在24GB GPU上服务96K context、在48GB GPU上服务>12× baseline batch size，同时保持低延迟和高吞吐。
