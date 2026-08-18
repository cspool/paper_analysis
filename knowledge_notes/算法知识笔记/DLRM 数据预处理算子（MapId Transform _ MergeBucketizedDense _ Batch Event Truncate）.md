## DLRM 数据预处理算子（MapId Transform / MergeBucketizedDense / Batch Event Truncate）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 数据预处理算子把原始特征转换为模型可输入表示，训练时在分布式 worker 执行（Zhao 2022），推理时嵌入模型 module 同步执行、延迟直接计入端到端响应。广告推荐模型消费四类特征：dense（float32 连续属性）、sparse（list<int64> 分类 ID，变长需 jagged tensor）、weighted sparse（list<pair<int64,float32>>）、event-based（用户历史事件时间序列）。预处理变换三类：dense 归一化（BoxCox、Logit、one-hot + 线性缩放）、sparse 处理（top-k 截断可选排序、cryptographic hashing 映射 ID 到 embedding 索引、int64→int32 downcast）、特征派生（bucketize 连续值到分类 bin、多稀疏表集合操作、n-gram hashing），共 200+ 算子分支。三个代表性算子：(1) MapId Transform——把高基数稀疏 ID 重映射为密集连续整数（1-indexed 位置，未知映射为 0）：idx=bucketize(v,M)；clamp；若 M[idx]==v 输出 idx+1 否则 0。(2) MergeBucketizedDense（MBDT）——连续特征按每特征 border 列表批量 bucketization：Y_{f,i}=min{k | X_{f,i}<B_f[k]}，border 展平加 inf 哨兵、offsets 全局索引（例：feature0 值[0.1,0.4,0.8]→bin[0,1,2]，feature1 值[0.2,0.5,0.9]→[3,4,5]）。(3) Batch Event Truncate——嵌套 jagged tensor（outer_lengths 每用户事件数、inner_lengths 每事件属性数、values 展平属性数据）按 N 事件截断，跨多特征协调三层索引算术。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- pipeline 中位置：raw features → 预处理算子链（dense 归一化 + sparse 截断/哈希 + 特征派生）→ embedding lookup（hash 后 ID 查表）→ NN 阶段。MapId 具体计算（伪代码）：
```
# V: [B] 输入 ID；M: 排序 mapping 表
idx = torch.bucketize(V, M)                    # 二分找插入位置
idx = torch.clamp(idx, max=M.numel()-1)
mapped = torch.gather(M, 0, idx)               # 查映射值
out = torch.where(mapped == V, idx+1, 0)       # 匹配→1-indexed，未知→0
# 例：V=[100,300,500,200,999], M=[100,200,300,400,500] → out=[1,3,5,2,0]
```
- MBDT 计算：每特征 border 展平（inf 哨兵分隔）+ offsets 记录起点，并行时每值在自身 border 区间二分（或向量化计数，对 3-10 元素 border 数组 O(n) 优于 O(log n)），输出加 offsets 得全局唯一 bin 索引。Batch Event Truncate：保留每用户前 N 个事件的所有属性（跨所有 feature 同步截断），被丢弃属性从 values 移除、outer/inner_lengths 重算（例：User0 有 [1,0,2] 属性事件 3 个、截断到 N=2 时丢弃第 3 事件 2 个属性）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：PyTorch 参考用 torch.bucketize/clamp/gather/where、torch.compile 编译；MTIA 上这些 ATen 算子部分缺失（v2i：clamp.out/gather.out/sort.values_stable/all.all_out/_unique2 等；v3 仍缺 clamp.out/sort.values_stable/_unique2/unique_consecutive），导致 CPU 回退与分离式部署。KernelEvolve 生成 fused Triton kernel：MapId 把 4 算子融合为单 kernel（in-register 20 步编译期 unroll 二分、tl.where 无分支更新、coalesced block-parallel 布局），MTIA v2i 最高 4.07×、v3 最高 1.36×；MBDT 融合全流程（向量化计数、自适应 block size 64/128/256、寄存器驻留），v2i 2.94-9.25×、v3 2.31-3.09×；Batch Event Truncate 把逐 feature 串行循环改成单 launch 多 feature 并行 batched kernel，最高 14.5×、生产端到端 2×。这些算子低算术强度但决定部署架构，是"kernel 覆盖优先于 GEMM 优化"论点的依据。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta
