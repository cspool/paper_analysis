# Abstract

Large multimodal models (LMMs) demonstrate impressive capabilities in understanding images, videos, and audio beyond text. However, efficiently serving LMMs in production environments poses significant challenges due to their complex model architectures and heterogeneous characteristics across their multi-stage inference pipelines and modalities.

We present the first comprehensive systems analysis of two prominent LMM architectures, decoder-only and cross-attention, across six representative open-source models, revealing key systems design implications. We also present an in-depth analysis of production LMM inference traces, uncovering unique multimodal workload characteristics, including variable, heavy-tailed request distributions and bursty traffic patterns.

Based on these insights, we propose ModServe, a modular LMM serving system that decouples model stages for independent optimization and adaptive scaling. ModServe dynamically reconfigures stages and handles bursty traffic with modality-aware scheduling and autoscaling to meet tail latency SLOs while minimizing costs. ModServe achieves 3.3–5.5× higher throughput (leading to 25– 41.3% cost saving) while meeting SLOs on a 128-GPU cluster with production multimodal traces.

