## SIKE（超奇异同源密钥封装）与 Montgomery ladder 侧信道

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SIKE（Supersingular Isogeny Key Encapsulation）是基于超奇异椭圆曲线同源的候选后量子 KEM，属 NIST PQC 竞赛 4 轮候选（已被 Castryck-Decru 密码学攻击破解，但仍是侧信道研究标杆目标）。Cloudflare 的 CIRCL（Interoperable Reusable Cryptographic Library）库提供 SIKE 实现。SIKE-751 的私钥是 378-bit 二进制整数 m，解密过程用 Montgomery ladder 逐位处理：若第 i 位与第 i−1 位不同（m^i ≠ m^(i-1)），ladder 的 (i+1) 步产生零值、导致解密停滞（stall）、功耗下降；若相同则正常计算、功耗较高。这种"数据相关的功耗差异"正是 Hertzbleed 与本文 TimeGaps 攻击的利用点。
- 在本文中：TimeGaps 采集器在 CIRCL 以 300 个并发 goroutine、10 个随机 378-bit 密钥运行时，逐位猜测过程中收集 TimeGaps；m^i≠m^(i-1) 时（零值 stall、处理器运行在更波动频率环境）无 TimeGap 出现概率 95.48%，m^i=m^(i-1) 时（稳定运行）无 TimeGap 概率升至 96.14%——该区分足以恢复密钥。最终只需确定首位是 0 还是 1，把密钥搜索空间降为 2。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- SIKE 解密 pipeline（Montgomery ladder，一位处理）：
```
m = 378-bit 私钥（明文形式）
for i in 1..378:
    if m^i != m^(i-1):        # 相邻位不同
        P_zero ← ladder 步骤产生零值 → 算力 stall → 功耗下降 → 频率波动 → 更少 TimeGaps
    else:                      # 相邻位相同
        正常同源计算 → 功耗稳定 → 更多 TimeGaps（无 TimeGap 概率 96.14% vs 95.48%）
    攻击者统计每个位猜测窗口内"无 TimeGap"概率 → 反推相邻位是否相同 → 重构 m
```
- 该 pipeline 属于"数据相关功耗 → 频率/挂起时间 → 观测分类"的侧信道算法链，与一般 ML 推理 pipeline 不同：输入是密码学运算，输出是密钥位。TimeGaps 作为测量原语插入到攻击者观测层，不修改受害者算法本身。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：SIKE 实现来自 Cloudflare CIRCL（https://github.com/cloudflare/circl），Go 语言库，攻击时 spawn 300 goroutine 并发跑 10 个密钥制造系统负载；攻击者程序与其并发运行，在 i5-8259U 上经 SSH 访问（减少 iGPU 噪声）。使用场景：评估新侧信道（Hertzbleed、TimeGaps）的密钥提取能力，作为与 SIKE 密码学攻击对比的标准侧信道 benchmark。指标：每 bit 猜测的 TimeGap 分布、无 TimeGap 概率差、最终密钥恢复是否只需 2 次尝试。

- **TIDE 版（macOS/Apple Silicon）**：与 TimeGaps（x86 频率/挂起时间）不同，本文用 TIDE 中断计时原语采集 SIKE 解密时的频率相关计数器变化。实验设置：MacBook Air 2023（M3），CIRCL v1.1 以 300 个并发 goroutine 跑 10 个随机 378-bit 密钥；每次 bit 猜测收集 50,000 个 TIDE 计数器值，只保留超过 3000 万的样本（低于该值的更可能被噪声中断污染）取平均。结果：m^i=m^(i-1)（无 stall）时计数器平均 32,119,284；m^i≠m^(i-1)（Montgomery ladder 零值 stall、处理器低频运行）时平均 31,362,263——可区分即恢复密钥，搜索空间降到首位 0/1 两个候选。前置实验：空闲系统下约 40% 的计数器值落在 40,470,000–40,540,000，对应 M3 P 核 4.050 GHz 下 100 Hz 的定时器中断间隔（macOS 固定间隔定时器中断）。

涉及论文标题：
- TimeGaps Channels: Exploiting CPU Halted Time for Fun and Profit
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon
