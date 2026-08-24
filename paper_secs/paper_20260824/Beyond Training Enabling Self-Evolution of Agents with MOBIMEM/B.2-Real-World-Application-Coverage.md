# B.2 Real-World Application Coverage

To ensure comprehensive evaluation across diverse usage patterns, we tested the system with 25 mainstream applications covering Social, Shopping, Information, and Video domains. These applications represent the most commonly used mobile apps in real-world scenarios.

### B.2.1 Application Categories

The tested applications span four functional categories corresponding to typical user activities. Table [5](#page-19-0) lists the applications in each category.

#### B.2.2 Task Scenarios

We select 6 distinct task scenarios that represent common real-world multi-app workflows. These scenarios range from simple sequential operations to complex parallel coordination tasks. By varying the specific applications used within each

Table 5: Real-world applications tested across four functional categories.

<span id="page-19-0"></span>

| Category | Applications & Description                 |
|----------|--------------------------------------------|
| Social   | WeChat, QQ, Sina Weibo (instant messaging  |
|          | & social networking)                       |
| Shop     | Taobao, JD.com, Pinduoduo, Xianyu, Ele.me, |
|          | Meituan (e-commerce & services)            |
| Search   | Xiaohongshu, Zhihu, Toutiao, Dianping,     |
|          | Browser (info retrieval & tools)           |
| Video    | Bilibili, iQIYI, Tencent<br>Video, Youku,  |
|          | Douyin, Kuaishou (video streaming)         |

| Scenario           | Logic & Example Instruction                 |
|--------------------|---------------------------------------------|
| search+shop+social | Serial/Pipeline:<br>Find<br>recommended     |
|                    | 2025 Canon cameras on Xiaohong              |
|                    | shu, search on Taobao, send details via     |
|                    | WeChat.                                     |
| multi-video+social | Parallel Query: Check iQIYI, Tencent        |
|                    | Video for Joy of Life 3 updates, notify via |
|                    | WeChat if found.                            |
| multi-shop+social  | Parallel Comparison: Query DJI Ac           |
|                    | tion 5 price on Taobao and JD.com, send     |
|                    | comparison via WeChat.                      |
| single-shop+social | Simple Pipeline: Query DJI Action 5         |
|                    | price on Taobao, send result via WeChat.    |
| search+social      | Info Sharing: Search Disney Christmas       |
|                    | event dates on Xiaohongshu, send sched      |
|                    | ule via WeChat.                             |
| search+shop        | Decision & Action: Find best-rated          |
|                    | Sony headphones under 1000 CNY on           |
|                    | Zhihu, order on Taobao.                     |

<span id="page-20-0"></span>Table 6: Task scenarios representing real-world usage patterns.

category and adjusting task parameters, we create a total of 50 test instances covering diverse usage patterns.

Table [6](#page-20-0) describes the task scenarios and provides example instructions.

#### B.2.3 Test Instance Construction

To ensure the evaluation reflects realistic usage diversity, we construct the 50 test instances according to the following principles:

Application Variation: For each task scenario, we randomly selected compatible applications within each category to ensure the agent handles different UI designs and interaction patterns. For example, shopping tasks alternated between Taobao, JD.com, and Pinduoduo.

Parameter Diversity: We varied search keywords, product names, target prices, and contact names across different instances to prevent result caching and ensure genuine task execution.

Cross-Category Coordination: We included tasks linking Video and Social apps (e.g., Douyin + QQ), Shopping and Search apps (e.g., Taobao + Zhihu), and Information Retrieval and Shopping apps (e.g., Xiaohongshu + Taobao) to test the scheduler's ability to handle data passing between diverse application architectures.

## B.3 Evaluation Methodology

For each task instance, we execute the workflow using all three execution modes. The primary metric is the end-toend latency (seconds), measured from the moment the user issues the command until the final action (e.g., message sent or order placed) is confirmed by the system. We calculate the speedup of the parallel modes relative to the serial baseline to quantify the efficiency improvements provided by the Agent

Scheduler. Each task instance is executed multiple times to ensure measurement reliability, and we report the average latency across runs.