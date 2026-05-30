# 8 Related Work

We briefly overview the state-of-the-art approaches to automatically scaling LLMs. In addition, we discuss their limitations and compare them with the proposed solution.

#### 8.1 Instance Replication and Scaling

Many current works incorporate horizontal scaling [\[6,](#page-12-8) [8,](#page-12-9) [18\]](#page-13-5), which involves adding or removing entire LLM serving instances to adapt to workload characteristics. This method benefits from simplicity, built-in fault tolerance, and isolation. In contrast, vertical scaling [\[18,](#page-13-5) [21\]](#page-13-6) adjusts resource allocation to a single instance, which grants fine-grained scaling at the cost of implementation complexity and service downtime. CoCoServe [\[21\]](#page-13-6) proposes a vertical scaling approach by replicating a subset of decoder layers onto underutilized hardware. Their work incurs some downtime during scaling and does not inherently support TP or EP, which are required to support larger models that cannot be stored on a single compute device. In contrast, we scale all layers, offer zero downtime during scaling, and support TP and EP.

#### 8.2 Efficient Scaling

A line of work [\[3,](#page-12-10) [6,](#page-12-8) [7,](#page-12-11) [14,](#page-12-12) [28\]](#page-14-2) has aimed at optimizing resource utilization and minimizing cold start latency. For instance, SpotServe [\[14\]](#page-12-12) and Llumnix [\[19\]](#page-13-7) reduce scaling overhead by lowering the cost of task migration across instances. BlitzScale [\[28\]](#page-14-2) and Scale [\[26\]](#page-14-1) leverage high-speed networks between GPUs to multicast model weights when adding new model instances to improve parameter loading times. Tetris [\[12\]](#page-12-13) identifies tensor redundancies in serverless inferencing and propose allowing function instances to share model weights and tensors, which improves memory efficiency. Despite these, scaling time remains a challenge, and many practical deployments rely on overprovisioning [\[27\]](#page-14-3), which ensures responsiveness but significantly increases operating costs.

## 8.3 MoE-Specific Scalability

A few complementary lines of work specifically address MoE models at scale. The first of these addresses the problem of suboptimal expert placement across devices, which results in poor MoE performance due to imbalanced load over the devices. MoEShard [\[5\]](#page-12-14) is an inference system that achieves optimal load balance by sharding MoE experts. Other works [\[4,](#page-12-2) [22\]](#page-13-3) look at replicating experts onto hardware based on their usage to better balance workload at the cost of additional memory.

Another recent topic involves disaggregating the attention modules from the MoE [\[17,](#page-12-15) [23,](#page-13-4) [29\]](#page-14-4), which allows independent scaling of each. However, to maximize performance and utilization, the modules need to be scaled in specific ratios. In addition, they do not describe any autoscaling functionality.

