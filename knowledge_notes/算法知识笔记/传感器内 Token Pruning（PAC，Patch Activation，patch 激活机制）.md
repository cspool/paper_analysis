## 传感器内 Token Pruning（PAC，Patch Activation，patch 激活机制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PAC 是 DESSCam 的传感器内 token 剪枝机制：把像素阵列按 N×N（16×16）分组为 patch，每个 patch 配一个 PAC 电路用列级加法器 + 加法树累计事件数，只有当累计事件数超过配置阈值（DESSCam 取 2）时，patch 才被激活并经握手读出——只有高事件密度的 patch 成为送入 ViT 的 token。其本质是 token pruning 的前移：把"对稀疏 token 序列做剪枝"从 host NPU 前移到像素阵列内，避免空间孤立冗余事件产生无效 token 和无效计算（最多减少 61% host 端算法 MAC），同时降低接口传输数据量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 每个 event frame：
for patch p in 22×17 grid:
    cnt(p) = adder_tree(event_flags of p)   # 16×16 事件计数
    if cnt(p) > TH:                          # TH=2
        p.active = True; handshake(p)        # ReqX/ReqY
        emit AER packet(p)                   # addrX/addrY + 512bit 事件 + 时间戳
tokens = [p for p.active]                    # 稀疏 token 序列
if len(tokens) >= 12: run ViT inference      # 12 patch 触发一次 gaze 估计
```
PAC 激活频率实测 162.76–64,170.78 Hz，12 patch 触发一次推理 → 等效帧率 13.56–5,347.56 Hz（延迟 0.19–73.75 ms 自适应）。与 Event Transformer（ICIP 2022）等 off-sensor token 剪枝对比：PAC 把 token 稀疏性计算放在像素阵列内，off-sensor 算法只需处理已剪枝的 patch。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现：每 16×16 像素共享一个 PAC 握手控制单元与加法树（patch 级握手替代标准 DVS per-pixel 握手，省复杂仲裁逻辑、输出延迟从 120 ns 降至数 ns）；算法等效：对 patch 序列施加稀疏掩码。与 BlissCam 片内 ROI 预测 NPU 对比：PAC 用简单计数电路实现 token 稀疏性，不需要片内 NPU，数字功耗更低。使用场景：事件相机输出天然稀疏、空间孤立噪声事件多的任务（眼动追踪），阈值可按噪声水平配置。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking
