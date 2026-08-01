# 多模态推理延迟优先：正式结果索引（v6 / Round 12）

本页是正式审计快照的可读结果索引。完整 W01、R01、Task、Context、来源 Ref 和
Runtime 记录位于 `run_snapshot.tar.gz`。源运行真实状态为 `PAUSED`，因此这里不把
结果声明为工作流最终 `FINISHED` 报告。

## 已接受 Anchor

1. `anchor-bc839d5a-400b-4861-87f8-a1c918e361cb`：任意模态混合路径的部署平衡
2. `anchor-f04dd75a-d81c-4257-8b64-3f897efb0432`：多模态异构链的主机回跳税
3. `anchor-1bd84d03-db20-4905-b68d-47d454f0a2e7`：自适应视觉长度触发的 GEMM 波量化悬崖
4. `anchor-313c69ad-dd0a-4d65-b07d-d8f1bfe68921`：视频 VLM 片上流式冗余浓缩架构

## 已接受 Direction

1. `direction-b3a2f39f-5938-406f-979a-db1aa40d0de7@2`：确定性瓶颈对齐的预算中性执行器再分配
2. `direction-c13ee353-cf5d-4856-9f31-8976b8601bba@2`：主机路由下的源侧 DRX 重整
3. `direction-2566ccdf-0e77-434d-a5fb-340aca73ba16@1`：单一 Triton backend 内的波感知双层 tile 调度
4. `direction-7737578b-0011-4e4a-958a-fa060871ceaa@2`：双估计量 SEC-only 提示感知流式裁剪

## 尚待修订

- `anchor-a2bae4dd-9f74-4cd9-af0b-86b705d977cd@1`：异构模态模块的批量—SM 争用
- Reviewer 要求：不能把 `22.2 → 34.4 requests/s` 归因于单一调度变化；现有
  证据把调度、预处理重叠和缓存复用捆绑在同一联合包中。
- 最新 Decision guidance：将该吞吐变化限定为联合包级观察；冻结缓存后的调度
  张力只依据异构 batch 曲线、低 SM 利用率和错误资源分配退化表述。

## 冻结时闭合状态

- 动态 6L coverage：L1–L6 均至少被一个当前 Anchor 覆盖；
- 对象局部 open query gaps：无；
- Runtime transport failures：无；
- 唯一机械阻塞：对上述待修订 Anchor 获得 Reviewer `PASS`；
- 此后仍需由 Decision 基于新的可信有界收敛探测判断是否可以完成开放探索 Goal。

该索引不替代压缩包内的原始 JSON。任何正式引用都应同时保留
`OFFICIAL_SNAPSHOT.md` 中的压缩包 SHA-256。
