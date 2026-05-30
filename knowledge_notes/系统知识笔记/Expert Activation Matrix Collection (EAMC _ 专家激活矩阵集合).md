## Expert Activation Matrix Collection (EAMC / 专家激活矩阵集合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Activation Matrix Collection (EAMC) 是 MoE-Infinity 中用于在线保存历史请求的 request-level EAM (rEAM) 的数据结构集合，充当激活模式匹配的"记忆库"。EAMC 具有固定容量（实验中 120 个 EAM 即覆盖 290 个 LLM 任务的大部分激活模式，仅占请求总数的 3%）。当新请求完成时，其 rEAM 被写入 EAMC；若 EAMC 已满，则替换与当前 rEAM 余弦距离最小的已有 EAM（维持多样性，优先保留差异大的模式）。EAMC 的容量由球覆盖理论（Sphere Covering）给出上界：75% 余弦相似度需 $2LE$ 个 EAM，98% 需 $\frac{1}{2}LE \ln(LE)$ 个 EAM。对于 $E=8\sim128$、$L=24\sim64$ 的主流 MoE 模型，最多约 40K EAM（160MB 内存）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
EAMC 在 MoE-Infinity 中的完整运转流程：

```
# EAMC 生命周期

[请求到达] → Prefill 完成 → rEAM_prefill 累积
           → Decode 每次迭代:
               iEAM_current → PredictEAM(iEAM_current, EAMC):
                 1. 将 iEAM_current 展平为向量
                 2. 与 EAMC 中所有历史 rEAM 计算 cosine distance
                 3. 返回距离最小的 top-N 个匹配 rEAM
                 4. 聚合匹配 rEAM → 行归一化 → pEAM
                 5. 施加 layer proximity decay: p[i][j] *= (1-(i-l)/L)
               → pEAM 用于 cache eviction + prefetching

[请求结束] → rEAM_final 归一化
           → if |EAMC| < capacity:
               EAMC.append(rEAM_final)
             else:
               # 替换策略: 替换与 rEAM_final 最相似的已有 EAM
               # 目的: 保持 EAMC 中激活模式的多样性
               most_similar = argmin(cosine_distance(rEAM_final, e) for e in EAMC)
               EAMC[most_similar] = rEAM_final

[故障恢复] → EAMC 可与 MoE checkpoint 一起 checkpoint
           → 重启后重新加载 EAMC，恢复 prefetching/caching 性能
```

EAMC 设计的两个关键洞察（来自论文 trace study）：
1. **请求内分组**：K-means 聚类显示 rEAM 可聚为 10-30 个组，组内 EAM 欧氏距离接近，意味着一个 EAM 可代表组内其他 EAM 的激活模式。
2. **跨请求转移不可预测**：组间 Markov 转移概率 <0.3（最高 ~0.3，大多 <0.12），因此基于学习的跨请求预测不可行，应通过匹配而非学习来做预测。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **实现**：MoE-Infinity 在 CPU 侧用 Python 列表存储 EAMC，每个 EAM 被展平为 $L \times E$ 维向量。匹配使用 FAISS 类库的矩阵乘法（Douze et al., 2024），测量 cost 为 21μs/query @1K EAMs、226μs/query @10K EAMs。EAMC 容量固定（默认从数百到数千），根据球覆盖理论，对于 $E=128$、$L=64$ 的模型，98% 余弦相似度需约 40K EAM（160MB）。
- **使用场景**：适用于 batch_size=1 的个人机器 MoE serving 场景。对于多请求批处理（cloud serving），由于 expert 激活趋于均匀分布，EAMC 匹配的有效性会下降。对于 workload shift（如从 MMLU 切换到 BIGBench），EAMC 平均在 30 个请求内恢复低延迟（<0.1% of dataset），因为相似任务共享激活模式。
- **局限**：论文指出聚类算法（如 K-means on EAMs）可进一步优化 EAMC 的组织和检索效率，但面对 Arctic-128x4B + FLAN 数据集（超 100 万 EAM，每个 4480 维向量），现有聚类库（包括 FAISS）无法高效处理，留作 future work。

涉及论文标题：
- MoE-Infinity Activation-Aware Expert Offloading for Efficient MoE Serving

---
