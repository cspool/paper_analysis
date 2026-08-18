## 空循环计数（loop counting）与 LSTM 网站指纹分类 pipeline

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 空循环计数（loop counting）是浏览器环境（无 rdtscp、无高分辨率定时器）下的执行速度测量原语：在固定时间间隔内数"空循环能完成多少次迭代"，迭代数越低说明期间发生过 TimeGap（CPU 挂起导致吞吐下降）。源自 bigger-fish（Cook et al., ISCA 2022，本文引用 [8]）的执行速度侧信道。本文验证：TimeGaps 总时长与 loop counter 值的 Pearson 相关为 −0.70±0.06（强负相关），比 Fish and Chips 报告的 loop counter 与中断处理时间相关 −0.49±0.11 更强，说明 TimeGaps 是 loop counter 凹陷的主要贡献者。
- LSTM 网站指纹分类 pipeline：以 500μs 间隔记录三种通道（native TimeGap 总时长 / 浏览器 loop counter / CPU 频率 scaling_cur_freq），把时间序列输入 32 单元 LSTM（与 prior work [8,44,57] 相同架构与超参），10 折交叉验证（81% 训练/9% 验证/10% 测试）输出 top-1 网站分类。数据集：Alexa Top 150 中前 100 个活跃非成人网站（sites/closed_world.csv），Chrome 每站 100 条 ×15s trace、Tor 每站 30s。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 检测与分类 pipeline：
```
# 采集阶段（每 500μs 一个时间步）
for each 500μs window:
    native:   gaps[i]   = 窗口内 TimeGap 总时长（rdtscp + SegScope）
    browser:  counter[i] = 窗口内空循环完成迭代数（JS）
    freq:     freq[i]   = 读 scaling_cur_freq
# 分类阶段
X = [gaps / counter / freq] 序列（T 时间步）→ LSTM(32 units) 逐时间步 h_t ← LSTM(x_t, h_{t-1})
→ softmax(100 类) → 10 折 CV 平均 top-1 准确率
```
- 例子（固定频率 Chrome）：native TimeGaps 序列 → 92.2±0.7% top-1；同一数据源在默认 DVFS 下 98.0±0.9%（DVFS 增加频率波动信息）；loop counter 浏览器通道固定频率 92.3±0.7%；频率通道固定频率下仅 ~1%（固定频率后频率不再变化，只有 TimeGaps 仍在泄漏）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：native 采集器 C（gaps_collector 系）+ SegScope；浏览器用 JS 空循环；分类用 Python（scikit-learn Random Forest 简化版 / 论文用 LSTM）。artifact 开源 Zenodo https://doi.org/10.5281/zenodo.19450827（MIT），attacker.py 自动化浏览器控制（Selenium+ChromeDriver）、数据采集与准确率评估；参数 --sites_list alexaN / --trace_length / --num_runs / --attack all / --core N，fixed_freq.sh/default_freq.sh 切换频率模式。使用场景：网站指纹攻击（浏览器场景攻击者只需让受害者访问恶意页面）、也可用于评估指纹防御（随机化防御下 TimeGaps 仍 83.1±1.3% vs 中断 61.2%）。

- **TIDE 版（本文，macOS/Apple Silicon）**：不依赖定时器，把 TIDE 中断计时 trace 直接喂给相同架构的 LSTM(32 units) 分类器（10 折 CV，训练 81%/验证 9%）。数据集：closed-world = Alexa top 100 网站×每站 100 条 trace；open-world 另加 Alexa top 1M 中 2000 个网站各 1 条（other-class）；视频 = YouTube 美国 top 20×每视频 200 条 trace，默认在 MacBook Air M3（4E+4P）采集。结果（表 III）：closed-world top-1 93.8%/top-5 98.7%，open-world 91.5%/98.5%，视频 78.1%/97.9%；用固定 24 MHz cntvct_el0 替换 TIDE 计数器（去频率缩放影响）后 93.1%/91.1%/87.2%。对比 Cook 定时器版（94.8%/74.5%）：TIDE 稍低但**在随机定时器防御下仍有效**（定时器版被打回 ~1%/5%）。跨硬件：10 网站 closed-world top-1 96.9%（M3 Air）/93.3%（M1 Pro 2021）/80.1%（M3 Max 2023）；双 TIDE 线程并行采集收益有限（95.7%/69.3%/74.2%），因为 Apple 均匀投递使每核信号变弱；播放 60s 视频做背景噪声时仍 60.6%。
- **loop-counting 增强（基于反推结果）**：因为 Apple 按 active core 数决定 SPI 投递，计算密集线程（不产生中断）会扭曲 trace（图 8）：训练/测试都无噪声 94.8%；训练无噪声+测试有噪声 39.3%；以"网站×active core 数"为联合标签（只改分类器最终层）后，有/无噪声分别 92.8%/93.6%（表 IV）。

涉及论文标题：
- TimeGaps Channels: Exploiting CPU Halted Time for Fun and Profit
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon
