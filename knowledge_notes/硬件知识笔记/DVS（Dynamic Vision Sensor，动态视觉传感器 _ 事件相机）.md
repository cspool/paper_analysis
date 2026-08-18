## DVS（Dynamic Vision Sensor，动态视觉传感器 / 事件相机）

术语解释
DVS 是像素级异步、事件驱动的图像传感器；每个像素独立感知对数光强变化，变化超过阈值才输出 ON/OFF 事件。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DVS 像素链：光电二极管 → 对数放大器 → 变化检测 → 两个比较器（VH/VL 阈值）→ 事件输出。与帧式传感器（固定曝光、全局帧读出）不同，DVS 无曝光概念：光强变化即时触发事件，时间分辨率微秒级，事件率与场景运动速度成正比（运动快事件多、静止几乎无事件）。在眼动追踪中的价值：快速眼跳（>300°/s）时帧式传感器需 kHz 帧率才能无模糊跟踪（1 kHz 追踪需 96 W 系统功耗），而 DVS 天然自适应（DESSCam 等效帧率 13.56–5,347.56 Hz 随眼动速度变化）。代价：像素 always-on，比较器持续耗电——实测比较器占 DAVIS346 单像素功耗 98.4%，DVS 像素功耗可达帧式像素 10×，低事件活动下整机仍耗几十 mW；另有背景活动噪声与热像素问题（泄漏大、持续误触发的像素）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DESSCam 对 DVS 像素的三处电路改造（SSPL 稀疏采样逻辑）：
```
标准 DVS：Vdiff -> 比较器(常开) -> 事件 -> per-pixel 握手 -> 高容负载事件总线（输出延迟~120ns）
DESSCam：Vdiff --SCtrl 门控--> 比较器 -> 2-bit SDP SRAM 锁存(EventFlag) -> PAC 加法树计数 -> patch 级握手（延迟数 ns）
```
① 1-bit 6T SCtrl SRAM 门控比较器：SCtrl=0 时切断比较器供电（比较器+采样电路分别省 12.6×/9.6×），仅 2% 使能像素产生动态功耗，像素功耗比标准 DVS 低至 5.26×；② 2-bit 8T SDP SRAM 就地锁存事件 + 全局 Spulse（200 ns）复位，免 per-pixel 握手、比较器不驱动高容事件总线；③ patch 级握手（16×16 共享 PAC）替代 per-pixel AER 握手。顶层 PSC 模拟前端经 hybrid bonding 传 Vdiff 到底层 SSPL。功耗模型：P_in_sensor = P_pixel_static + (1/M)Σ N_ev(s)·E_ts/T_acc(s) + P_logic（静态/动态/数字逻辑三段）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
商用实现：iniVation DAVIS346（346×260，DESSCam 分辨率对齐）、Sony IMX636/EVS（1280×720 堆叠）、Samsung 640×480/1280×960；事件率可达数 kHz 帧率等效。使用方式：机器人/高速视觉/眼动追踪传感器前端；AER 包（地址+极性+时间戳）经 MIPI CSI-2 或 USB 输出。设计要点：比较器功耗是像素功耗主体（门控/稀疏使能是关键省电手段）、握手仲裁延迟是输出延迟主体（patch/行级聚合是关键降延迟手段）。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking
