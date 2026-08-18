## WQE（Work Queue Entry）与流控字段（wqe_sync/fence/rx_sync/sync）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WQE（Work Queue Entry，工作队列条目）是 MTIA 300（ISCA'26）HCCL 集体通信在设备端的执行单元：collective 被翻译为 subgraph，subgraph 表示为 WQE 数组，每个 WQE 描述一类操作并直接映射到 RDMA work request（SEND/RECV/WRITE）或本地动作。WQE 类型：**SEND/RECV/WRITE**——与 RDMA WR 同语义（queue pair ID、本地/目标地址、长度、lkey/rkey 等）；**SET**——向本地内存（HBM 或 cache）写一个值；**WAIT**——阻塞直到某内存位置的比较器满足（如 wait 地址 0xabcdef > 10，可视为内存级条件变量）；**REDUCE**——执行和操作 S=A+B（S 可与 A 或 B 重叠，或可选做内存拷贝、充当 DMA 引擎）。流控字段定义 WQE 间顺序，支撑 ring/recursive doubling/ordered tree 等通信模式：**wqe_sync**（本 WQE 等到指定前序 WQE 完成才发出）、**fence**（本 WQE 完成后才发任何后续 WQE）、**rx_sync**（等所有 outstanding receive WQE 完成）、**sync**（等所有前序 WQE 完成）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次 4 节点 AllReduce ring 的 WQE 映射（ReduceScatter + AllGather 两阶段，见 AllReduce Ring 条目）：
```python
# ReduceScatter 阶段（每节点处理 1/N 分片，N=4）
wqes_rs = []
for step in range(N-1):                       # 3 步
    wqes_rs.append(SEND(buf=chunk[k], to=next))          # 并行可发
    wqes_rs.append(RECV(buf=recv_chunk, from=prev))      # 无依赖可并行
    wqes_rs.append(REDUCE(S=A+B, dst=local_chunk))       # 依赖前 RECV（wqe_sync）
    # 归约完成再解阻塞下一轮 RECV/SEND（wqe_sync 回指）
# AllGather 阶段（每步依赖前一步数据搬移）
for step in range(N-1):
    wqes_ag.append(SEND(gathered_chunk, to=next))        # 依赖上一 SEND 完成
    wqes_ag.append(RECV(buf=gathered_chunk, from=prev))  # sync 保证顺序
```
执行：WQE 顺序发出但仅按流控字段阻塞；subgraph 间逻辑并行（硬件可用性排队）；16 ME 并发多 subgraph。作用：把 ring/树等算法编码成数据依赖图，让 NMC/ME 在无主机参与下自动流水。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：HCCL 先验（a priori）按 outstanding work/拓扑/通信类型选算法与通道 → 生成 work packets/subgraphs/WQEs → 经 MTIA streaming interface 提交、ME 的 CPU-M 执行（共享 CQ 收 WQE）；WQE 的内部字段（流控字段 + RDMA 字段）在 ME 硬件执行。使用场景：AllReduce/ReduceScatter/AllGather/AllToAll 的 ring/recursive doubling/ordered tree 模式；SET/WAIT 还用于 subgraph 间依赖（内存比较器同步）。与 GPU 对照：NCCL 的通信由主机驱动 kernel 与 ring buffer，WQE 模型把整个 collective 变成设备端数据依赖图。信息缺口：论文未给出 WQE 的内存布局（位宽/字段编码）。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance

3DGS 补充视角（ISCA'26，GPU radix sort 作为 3DGS 渲染排序 baseline）：gsplat[51] 库的 3DGS 渲染管线用 GPU radix sort（Merrill & Grimshaw PACT 2010 [31]）做 tile 内 Gaussian 深度排序——把 (tile_id, depth) 打包为 64-bit 键后按键排序，得到按 tile 分组、组内深度升序的 Gaussian 列表再 α-blending。本论文把深度排序整体替换为 MLP-OIT（cuBLAS GEMM 直接输出 F(d_i)，跳过排序）：GPU 上因 MLP 算术强度低（1 深度参数仅 6 MAC vs 光栅化每 GS 256×6 MAC，约 30 倍差）而 memory-bound，几何均值延迟为 radix sort baseline 的 1.59×（更慢），论证 GPU 上排序仍更优；专用加速器上 MLP-OIT 相对 32 并行 bitonic 排序网络 21.1~32.4× 加速。与 MTIA 300 的硬件 radix sort（SFU 内桶化+直方图，用于 embedding 反向索引重排）用途不同：此处是通用 GPU 库排序 kernel 作为渲染管线 baseline。
