## ModifiedShared（MS）缓存状态（脏行同簇共享状态）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MS 是 Dorado 在 MESI 上新增的缓存状态：远端 home 的脏行被本地簇多个核共享时，这些核的行都处于 MS 状态。动机：核写远端行后在本地建 D=1 的 Temporary home 条目；若第二个本地核读该行，传统做法要把脏数据传回 Global home 更新内存并把两处目录项改 D=0（一次远端回写）；Dorado 改为保持 Temporary home 条目 D=1，两核都持脏行（各用 RLptr 登记）并转 MS 状态。与 MOESI 的 O（Owned，脏共享、单一 owner 负责写回）不同，MS 允许多个共享者同时持脏行、Global home 完全不被更新（仍只有 LRptr + D=1）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转例子（Table IV 的 B4 / Table VI 的 D4）：c1 写远端行→本地 Temporary home 条目 D=1（RLptr 指 c1，c1 持 M）；c2 读→本地条目保持 D=1、加 RLptr 指 c2，c1、c2 都转 MS，不通知 Global home；c3 再写→无效化所有 MS 共享者、自己转 M，同样不通知 Global home（写回留待行驱逐或所有权转移）。收益：省掉脏行的跨簇回写与目录状态往返，把写-读-写序列完全留在簇内。TLA+ 验证性质：MS 行只能与 MS 或 I 状态的行共存（单写者多读者语义扩展）；脏位一致性要求 M/MS 行的 Global home 与本地 Temporary home 条目 D 位均置 1。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现（论文）：MESI 协议表扩展（描述用 MSI 简化、评估用 MESI+MS）；状态集 {M,E,S,I,MS} 纳入 TLA+ 模型检验。使用要点：适用于"簇内多核共享脏行"的局部性场景；与 Dir2B 等基线（plain MESI）相比不增加状态硬件复杂度（1 个额外状态编码）。论文未明确说明 MS 与 MOESI O 在写回责任上的进一步差异。

涉及论文标题：
- Dorado: Clustered Hardware Cache Coherence for 1,000+ Cores
