## libsmctrl / Green Context（CUDA 流到 SM 子集的硬件绑定）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
libsmctrl 是底层 CUDA 库，运行时修改 CUDA Stream 的 SM（Streaming Multiprocessor）执行掩码：调用 libsmctrl_set_stream_mask() 把某 stream 上后续 kernel launch 限制在 GPU 特定 SM 子集，实现进程内 SM 空间分区。Green Context（green-ctx）是另一套 SM 配额机制（上下文级方案），两者都是"把 CUDA 流/上下文绑定到指定 SM 集合"的运行时支持。libsmctrl 通过直接修改 CUDA stream 内部 metadata（GPC 配置掩码）工作：stream 上发射 grid 时 GigaThread Engine 按 mask 决定把 thread block 分发到哪些 SM；改 mask 后已排队后续 kernel 自动遵从新 mask，无需重建 stream/context，更新开销 ~4us（相对 GreenContext 的 context 切换需重初始化 CUDA Graph 等资源更便宜）。RESONATOR 用它实现 Intra-GPU Sharing 的 wide/narrow stream SM 配额与 SM 分区。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
libsmctrl 绑定流的 kernel 执行流程（RESONATOR 场景）：
```
gpc_info = libsmctrl_get_gpc_info()          # 获取 GPU GPC/TPC 拓扑
# 构建 SM mask（A100: 108 SM，以 16 SM 为粒度）
mask_wide  = build_sm_mask(all_sms)           # wide 流：全部 SM
mask_narrow = build_sm_mask(sm_ids[0..q_narrow*SM_total])  # narrow 流：窄子集
libsmctrl_set_stream_mask(s_enc_wide,   mask_wide)
libsmctrl_set_stream_mask(s_enc_narrow, mask_narrow)
libsmctrl_set_stream_mask(s_llm_wide,   mask_wide)
libsmctrl_set_stream_mask(s_llm_narrow, mask_narrow)
# 运行期（contending 模式）：每 kernel 查 profile 表选流后 launch
LaunchOnStream(k, s)   # s 已是绑定了 SM mask 的流，kernel 自动只在 mask 内 SM 执行
```
Annotations：RESONATOR 的 SMCTRL.SetQuota(stream, quota) 抽象在 libsmctrl 层实现为 set_stream_mask；complementary 模式（decode 保护）也是同机制：decode 流绑 SM_dec 切片、encoder 用其余 SM；依赖特定 NVIDIA driver 版本兼容性（libsmctrl 用未文档化内部 API）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：用户空间库（Bullet 的修改版 https://github.com/zejia-lin/BulletServe；论文引 [26] Bakita & Anderson RTAS'23 的硬件计算分区）。使用：创建多条 CUDA 流、各自 set_stream_mask 后按需发射 kernel，适合"一卡内分时/分空承载多个负载"的 serving 场景——RESONATOR 的 encoder+LLM 共存（wide/narrow 双流每任务）、Bullet 的 prefill/decode SM 动态重分区（平均 4.1us repartition）。相对 MIG（多实例 GPU 硬件分区）粒度更细、切换更快；相对默认 SM 抢占式调度提供显式隔离/配额控制。

涉及论文标题：
- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
