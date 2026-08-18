## Vertex-centric 图处理模型（顶点中心 / think-like-a-vertex）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- "像顶点一样思考"的图计算编程模型（Google Pregel 首创，BSP 超步迭代；后续 Giraph/GraphX/PowerGraph、GPU 侧 Gunrock/Ligra 沿用）：每个顶点维护自身状态与邻接边，每轮超步并行执行同一顶点程序——顶点先处理出边产生发给邻居的更新消息，再读入上一轮收到的更新消息修正自身状态，直至无消息发出（投票停机）。论文语境：机器人图算法（BFS、单源最短路 SSSP，用于 MoveBot-PRM 的 Gunrock/Ligra）自然符合顶点中心模型——A*/BFS 维护 frontier 节点队列，本质是"顶点处理出边队列 → 邻居收更新"的两阶段迭代，两阶段都是队列操作，因此能直接映射到 Morpha Core 的 queue-centric SIMD（graph morpha 的 MIMD-over-SIMD：每核用自己的 SIMD 指令流处理分配的顶点子集，跨核顶点更新走共享队列 + REMOTE_STORE，顶点→核映射静态）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 通用 SSSP 顶点程序（Web：GraphScope 文档）：
  ```
  def VertexProgramForSSSP():
      msgs = ReceiveMessages()               # 上一超步的入边更新队列
      m = Reduce(msgs, MIN)
      if dist > m:
          dist = m                            # 更新自身状态
      for n in neighbors:
          if dist + w(edge) < n.dist:
              SendMessage(n, dist + w(edge))  # 发出边更新
  ```
- 论文给出的 Morphatron 指令版（processEdges 第一阶段）：
  ```
  INIT_Q q0, FALSE; INIT_Q q1, FALSE; INIT_Q q2, TRUE   # q2 为共享队列
  ADD edge_load_offset, offset_queue.PEEK(), edges_addr
  LD_Q q0, edge_load_offset, offset_queue.POP()          # 标量段装载出边
  SYNC code_block_end; SYNC exe_start, SIMD
  Q_LOOP_UNTIL_EMPTY q0, 3:
      POP_Q src_queue, src
      ADD new_val, src.val, q0.POP()
      PUSH_Q q1, (src.dst, new_val)                      # 更新打包入队
  Q_LOOP_UNTIL_EMPTY q1, 1:
      REMOTE_STORE graph, q1.POP(), q2                   # 写入邻居的共享入边队列
  SYNC exe_end
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 通用实现：分布式（Pregel/Giraph，消息传递 + 超步 barrier）或单机 GPU（Gunrock 用 frontier/advance 抽象、Ligra 用 push/pull 方向切换）。局限：每超步信息只传播 1 跳，power-law 图上收敛慢（Web 综述）。本文用法：把"顶点内计算"与"顶点间更新传播"都看作队列操作，落到 queue-centric SIMD 硬件；跨核经电路交换互连 + 共享队列完成，两阶段间全体核 standby 让互连充当更新写入的存储介质。

涉及论文标题：
- Accelerator Polymorphism: Transcending Domain-Specific Architectures with Robotics
