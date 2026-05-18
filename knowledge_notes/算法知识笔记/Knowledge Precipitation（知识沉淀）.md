## Knowledge Precipitation（知识沉淀）

术语是什么？通过联网搜索让回答具体和精准。
Knowledge Precipitation是MFS论文提出的一种离线fine-tuning方法，将LLM model family中最大的模型微调为一个嵌套式multi-tier模型，使低tier获得独立的语言建模能力，同时由高tier的梯度向低tier"沉淀"知识。核心思想是利用同一model family中模型结构（均为stacked decoder-only Transformer）的统一性和Transformer的layer/head冗余性，通过full-parameter fine-tuning在最大模型上将不同模型规模的能力折叠到单一checkpoint中，替代独立维护多个模型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Knowledge Precipitation的算法pipeline以Llama2-13B-chat转3-tier MFS模型为例：
1. Tier结构设计：测量最大模型（Llama2-13B）各层在目标推理硬件（A100）上的实际latency→确定tier边界（如tier-1取前18层对齐3B latency、tier-2取前32层对齐10B latency、tier-3为全部40层=13B）。切分原则：(a) 只切layer不切head，保持attention中所有head一致性和KV cache兼容性；(b) latency-aligned而非parameter-aligned，使各tier的实际用户体验对齐对应规模独立模型。
2. Step-by-step fine-tuning：
   - Step 1 (Tier-3)：对Llama2-13B-chat做全参SFT，loss仅含tier-3输出头L1 = L_tier3_output。数据为guanaco-llama2 ~9.85k对话，AdamW lr=2e-5, half-period cosine LR, weight decay=0.1, gradient clipping=0.3, 8×gradient accumulation (effective batch=64), seq_len=4096, 1 epoch/2500 iterations。
   - Step 2 (Tier-2)：基于Step 1 checkpoint，在第32层添加tier-2输出头（lm_head），训练目标L = L_tier2 + λ3·L_tier3。tier-3梯度反向传播到前32层共享参数，使tier-2获得独立生成能力且tier-3质量不退化。
   - Step 3 (Tier-1)：基于Step 2 checkpoint，在第18层添加tier-1输出头，训练目标L = L_tier1 + λ2·L_tier2 + λ3·L_tier3。低tier通过接收高tier梯度"沉淀"知识，获得独立语言建模能力。
3. 各tier有独立loss和输出头，但共享低层Transformer参数。推理时请求在对应tier边界采样返回（低tier早退出→低延迟低成本；高tier继续执行→高质量）。

伪代码（简化）：
```
# Knowledge Precipitation fine-tuning
model = load_checkpoint(llama2_13b_chat)
add_lm_head(model, layer=18, name="tier1_head")  # 3B-equivalent
add_lm_head(model, layer=32, name="tier2_head")  # 10B-equivalent
# tier3_head = original lm_head at layer 40  # 13B

# Step-by-step training
for step in [tier3_only, tier2_tier3, tier1_tier2_tier3]:
    for batch in guanaco_llama2_dataset:
        if step == tier3_only:
            loss = cross_entropy(model.tier3_head(hidden[40]), labels)
        elif step == tier2_tier3:
            loss = cross_entropy(model.tier2_head(hidden[32]), labels)
                 + λ3 * cross_entropy(model.tier3_head(hidden[40]), labels)
        else:  # tier1_tier2_tier3
            loss = cross_entropy(model.tier1_head(hidden[18]), labels)
                 + λ2 * cross_entropy(model.tier2_head(hidden[32]), labels)
                 + λ3 * cross_entropy(model.tier3_head(hidden[40]), labels)
        loss.backward()
        optimizer.step()
```
关键设计决策：(a) Step-by-step而非joint training——避免多tier loss梯度冲突导致低tier性能不可控；(b) 对较低tier使用较高λ权重以补偿其较小的模型容量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
论文在2×8 H800 SXM5 (80GB) GPU上实现Knowledge Precipitation，两台服务器通过8×400Gbps NDR InfiniBand互联。约24小时完成Llama2-13B的3-tier fine-tuning（2500 iterations）。论文受资源限制（16×H800）未实际fine-tune Llama2-70B级别模型，大模型家族规模上的成本外推仍需验证。论文未开源Knowledge Precipitation实现代码。该方法目前是MFS系统的专有技术，未见其他系统采用类似方法。通用化使用时需考虑：(1) 需要模型家族中最大的模型的checkpoint作为起点；(2) full-parameter fine-tuning成本随模型规模线性增长；(3) tier边界的latency测量需在目标推理硬件上进行。

涉及论文标题：
- MFS: An Efficient Model Family Serving System for LLMs

