# 4 ByteScale Overview

We present ByteScale to address these challenges. As shown in Figure 7, it consists of three main components. Profiler is to profile the environment, model configuration, data distribution, and build cost models for other components. Communication Optimizer is to improve the communication efficiency for both short and long sequences by data-aware sharding, dynamic communication, and selective offloading. Balance Scheduler is to solve the imbalanced computation by parallelism-aware data assignment.

