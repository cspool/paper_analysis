# 2 Background

#### 2.1 Serverless Inference Workflow

Cloud-based ML inference services have increasingly turned to serverless technology to streamline the deployment of the serving pipeline [10, 37, 47, 48, 50, 52, 54]. Serverless inference allows users to deploy ML models and data processing operations as stateless functions and lets the platform to handle resource provisioning, autoscaling, logging, fault-tolerance, and other infrastructure management tasks. Users are only billed when functions are running, eliminating the cost of idle resources.

Modern inference services typically orchestrate multistage workflows that integrate ML models with data processing operations. Fig. 1 illustrates this with a real-world traffic monitoring application [40], which comprises one CPU function for video decoding and five GPU functions for pre-processing, object detection, post-processing, and person and car recognition. These heterogeneous functions are loosely coupled, composing a serverless inference workflow. The diversity of these workflows is further demonstrated in Fig. 12, which collects a suite of real-world inference services from recent studies [3, 8, 11, 40, 54, 55]. The suite spans multiple workflow patterns, including linear sequential pipelines, conditional branching for dynamic decision-making, and fanin/fan-out parallelism for high-throughput data distribution.

