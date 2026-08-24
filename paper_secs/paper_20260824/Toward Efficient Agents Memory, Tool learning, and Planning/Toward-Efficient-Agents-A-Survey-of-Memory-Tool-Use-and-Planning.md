# **Toward Efficient Agents: A Survey of Memory, Tool Use, and Planning**

**Xiaofang Yang**1,2,**† Lijun Li**1,**†**, **Heng Zhou**1,3,**† Tong Zhu**1,**† Xiaoye Qu**<sup>1</sup> **Yuchen Fan**1,4 **Qianshan Wei**<sup>5</sup> **Rui Ye**<sup>4</sup> **Li Kang**1,4 **Yiran Qin**<sup>6</sup> **Zhiqiang Kou**<sup>7</sup> **Daizong Liu**<sup>8</sup> **Qi Li**<sup>5</sup> **Ning Ding**<sup>9</sup> **Siheng Chen**<sup>4</sup> **Jing Shao**1,

> Shanghai Artificial Intelligence Laboratory, <sup>2</sup> Fudan University, University of Science and Technology of China, <sup>4</sup> Shanghai Jiaotong University, 5 Institute of Automation, Chinese Academy of Sciences, The Chinese University of Hong Kong (Shenzhen), Hong Kong Polytechnic University, <sup>8</sup> Wuhan University, <sup>9</sup> Tsinghua University

## **Abstract:**

Recent years have witnessed increasing interest in extending large language models into agentic systems. While the effectiveness of agents has continued to improve, **efficiency**, which is crucial for real-world deployment, has often been overlooked. This paper therefore investigates efficiency from three core components of agents: **memory, tool use, and planning**, considering costs such as latency, tokens, steps, etc. Aimed at conducting comprehensive research addressing the efficiency of the agentic system itself, we review a broad range of recent approaches that differ in implementation yet frequently converge on **shared high-level principles** including but not limited to bounding context via compression and management, designing reinforcement learning rewards to minimize tool invocation, and employing controlled search mechanisms to enhance efficiency, which we discuss in detail. Rather than treating efficiency as a standalone objective, we adopt an effectiveness-first perspective that an agent is meaningfully efficient only when it preserves acceptable task quality. Accordingly, we characterize efficiency in two complementary ways: comparing effectiveness under a fixed cost budget, and comparing cost at a comparable level of effectiveness. From this perspective, we survey benchmarks along both dimensions. We first review effectiveness-oriented benchmarks across holistic and component-specific evaluations, and then examine how efficiency is measured in practice, summarizing evaluation protocols and commonly reported metrics from both benchmark papers and method-level studies. Moreover, we discuss the key challenges and future directions, with the goal of providing promising insights.

**Keywords**: Agents, Efficiency, Agent Memory, Tool Use, Planning

**Date**: January 20th, 2026

**Projects**: <https://efficient-agents.github.io/>

**Code Repository**: <https://github.com/yxf203/Awesome-Efficient-Agents>

**Contact**: [yangxiaofang@pjlab.org.cn,](mailto:yangxiaofang@pjlab.org.cn) [lilijun@pjlab.org.cn,](mailto:lilijun@pjlab.org.cn) [hengzzzhou@gmail.com,](mailto:hengzzzhou@gmail.com) [zhutong@pjlab.org.cn](mailto:zhutong@pjlab.org.cn)

<sup>†</sup> *Main contributors*

*Corresponding Author*

