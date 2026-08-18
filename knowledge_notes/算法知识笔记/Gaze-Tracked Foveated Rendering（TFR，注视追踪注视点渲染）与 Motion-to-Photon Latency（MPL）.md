## Gaze-Tracked Foveated Rendering（TFR，注视追踪注视点渲染）与 Motion-to-Photon Latency（MPL）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TFR 是 AR/VR 中利用眼动追踪的 gaze 方向数据驱动 GPU 只对注视点（foveal）区域做高分辨率渲染、周边区域低分辨率渲染的渲染负载削减技术（Tobii 在 Pico 头显实测渲染开销最多降 72%、平均降 60%）。MPL 是"从虚拟动作到其视觉反馈被感知"的端到端延迟 = 眼动追踪延迟 + TFR 与显示延迟；研究证明 MPL 需 <5 ms 才不引起视觉不适/晕动，而商用 HMD（如 Vive Pro Eye）MPL 高达 79 ms。眼动追踪延迟在商用系统占 MPL 的 63.3%、在 SOTA 研究中占 77.7%，是 TFR 能否落地的主要瓶颈；提高帧率降延迟会带来功耗激增（1 kHz 追踪频率需 96 W），形成延迟-功耗权衡。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
gaze 采样 -> gaze 估计 (x,y) -> 划分 fovea/周边区域 -> 分级分辨率渲染 -> 显示
MPL = t(眼动追踪) + t(TFR 渲染) + t(显示)   # 要求 < 5 ms
```
眼动追踪是 TFR 管线的前置级：其延迟决定注视点渲染跟随眼球运动的实时性（延迟过大则渲染区域落后于实际注视点、产生视觉伪影），其功耗决定 HMD 整机功耗预算（商用眼动追踪 >2W，接近 VR 系统功耗预算一半）。DESSCam 以亚 1 ms 眼动追踪 + 数 mW 功耗直接服务 TFR 部署。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
商用实现：HTC Vive Pro Eye、Tobii XR（帧式传感器，延迟 >50 ms）；研究实现：BlissCam（in-sensor 稀疏采样，8.2× 省电但 9.4 ms 延迟）、TinyTracker（IMX500 近传感器计算）、DESSCam（DVS 事件驱动 + ESS 稀疏采样，15.2× 延迟降低）。使用场景：HMD 渲染负载削减、超轻量智能眼镜（如 Meta Ray-Ban 154 mAh 电池、49.6 mW 全天功率预算）中把眼动追踪压到数 mW 级。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking
