## 侧信道攻击与指纹攻击（Side-channel / Fingerprinting Attack）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
侧信道攻击是攻击者观测不知情受害者的正常行为在共享资源上留下的附带可观测效应（时序、功耗、缓存状态等）来推断敏感信息，与隐蔽信道的关键区别是受害者不配合。指纹攻击（fingerprinting）是其应用形态：为每个目标工作负载建立独特的观测签名，再用分类器把新观测归类到目标。DarkStream 的侧信道：受害者进程经 Intel DTO 库把 libc 内存操作卸载到 DSA，攻击者进程在同一 DSA 设备持续提交 1 MB Memory Move 制造争用并逐请求记录延迟；不同网站/DL 模型的 DSA 操作频率与尺寸分布不同（如 youtube.com 约 6500 次 Fill + 800 次 Move，EfficientNet-B7 在 8–12 KB 段 124 次 Fill），调制出不同的延迟 trace 模式（基线约 75000 cycles、争用尖峰 >80000 cycles 的时长/频率模式）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
攻击流程两阶段：(1) 训练期——攻击者跑目标应用（Playwright+Chromium 访问网站 / DL 推理），同时在同一 DSA 上探测并记录每个目标的争用延迟 trace，为每目标积累训练集；(2) 推理期——受害者执行任意目标应用，攻击者同法采集 trace 输入分类器识别目标。数据管线：延迟 trace → 渲染 1200×800 像素图像（不带 URL/索引标签）→ ResNet-18 图像分类器（10 epoch）。网站指纹：43 个最常访问网站各 300 次 = 12900 样本（240/30/30 划分），另加 open-world 设置——Alexa Top 1M 中前 4000 个未监控网站各 1 次（训练/验证/评估各 2000/1000/1000），分类器多一个 'other' 类；结果总准确率 97.03%（20 个网站 100%，最低 63.3%，'other' 类 99.7%）。DL 模型指纹：12 种 CNN/ViT 架构（ResNet-50、DenseNet-161、EfficientNet-B7 等），每模型 200 次 = 2400 样本，8:1:1 划分，总准确率 99.17%（12 个中 10 个 100%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
关键工程选择：探测操作尺寸——隐蔽信道追求速率用 1-byte 最小传输，侧信道追求精度改用 1 MB：大传输让攻击者操作长时间占据共享数据通路，受害者干扰在长操作内累积放大（1 B/1 KB 探测的延迟方差大、无法区分受害传输尺寸，1 MB 下延迟与受害传输尺寸强正相关）；采集窗口——网站 7 s（足够完整加载一次页面）、DL 推理 10 s。软件栈：accel-config 配置 DSA、Playwright 驱动 Chromium、DTO 透明卸载（最低推荐设置：卸载 >8 KB 的操作）、ResNet-18（PyTorch）。Web 证据：IEEE 文献（"Mitigating timing based information leakage in shared schedulers"）对 side-channel"无受害者配合、观测附带泄漏"的经典定义。
- TimeGaps 补充视角（ISCA'26，CPU 挂起时间侧信道）：网站指纹的观测源从"共享资源争用"换成"CPU 频率切换的 halted 时间"——渲染不同网站产生不同 CPU/iGPU 工作负载 → 频率切换频率/幅度不同 → TimeGap 序列不同。采集三通道（native 每 500μs 的 TimeGap 总时长 / 浏览器空循环计数 / scaling_cur_freq 频率），用 32 单元 LSTM（10 折 CV：81% 训练/9% 验证/10% 测试）分类 Alexa Top 100 网站。结果：native TimeGaps 固定频率 Chrome 92.2±0.7%/Tor 87.4±0.9%，默认 DVFS Chrome 98.0±0.9%/Tor 85.2±1.0%（频率数据默认 DVFS 仅 93.3%/63.0%，固定频率下 ~1%——即固定频率下 TimeGaps 是唯一有效信号）；0.1s 窗口 TimeGaps 即达 Top-5 57.1%。与 DarkStream 的 DSA 争用信道互补：本文信道不需要共享加速器、不依赖 Intel DTO 卸载，且固定 CPU 频率后仍有效（iGPU 频率切换无法从用户态固定）；时序相关性显示页面加载时延与 TimeGaps 相关 −0.71±0.06，强于中断 −0.37±0.07 与频率 0.61±0.05，说明 TimeGaps 是比中断/频率更强的指纹信号源。

涉及论文标题：
- DarkStream: Exploiting Internal Throughput Contention in Data Streaming Accelerator for Timing Attacks
- TimeGaps Channels: Exploiting CPU Halted Time for Fun and Profit
