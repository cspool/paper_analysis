# **Abstract**

Large Language Models have revolutionized natural language processing, yet serving them efficiently in data centers remains challenging due to mixed workloads comprising latency-sensitive (LS) and best-effort (BE) jobs. Existing inference systems employ iterationlevel first-come-first-served scheduling, causing head-of-line blocking when BE jobs delay LS jobs. We introduce QLLM, a novel inference system designed for Mixture of Experts (MoE) models, featuring a fine-grained, priority-aware preemptive scheduler. QLLM enables expert-level preemption, deferring BE job execution while minimizing LS time-to-first-token (TTFT). Our approach removes iteration-level scheduling constraints, enabling the scheduler to preempt jobs at any layer based on priority. Evaluations on an Nvidia A100 GPU show that QLLM significantly improves performance. It reduces LS TTFT by an average of 65.5× and meets the SLO at up to 7 requests/sec, whereas the baseline fails to do so under the tested workload. Additionally, it cuts LS turnaround time by up to 12.8× without impacting throughput. OLLM is modular, extensible, and seamlessly integrates with Hugging Face MoE models.

### **CCS** Concepts

• Software and its engineering  $\rightarrow$  Software performance; • Computing methodologies  $\rightarrow$  Neural networks.

### **Keywords**

Large Language Models, Mixture-of-Experts, Preemptive Scheduling, Latency-Sensitive Inference, GPU Acceleration, Priority-Aware Scheduling

### **ACM Reference Format:**

Mohammad Siavashi, Faezeh Keshmiri Dindarloo, Dejan Kostić, and Marco Chiesa. 2025. Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference. In *Proceedings of 5th Workshop on Machine Learning and Systems (EuroMLSys '25)*. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3721146.3721956

\*Work done while at KTH Royal Institute of Technology

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for third-party components of this work must be honored. For all other uses, contact the owner/author(s).

EuroMLSys '25, Rotterdam, Netherlands
© 2025 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-1538-9/25/03
https://doi.org/10.1145/3721146.3721956

