# <span id="page-0-0"></span>Chameleon: Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments

[Nikoleta Iliakopoulou](https://orcid.org/0009-0000-0519-8683) University of Illinois Urbana-Champaign Urbana, USA nmi4@illinois.edu

[Tianyin Xu](https://orcid.org/0000-0003-4443-8170) University of Illinois Urbana-Champaign Urbana, USA tyxu@illinois.edu

[Jovan Stojkovic](https://orcid.org/0009-0007-4914-336X) University of Illinois Urbana-Champaign Urbana, USA jovans2@illinois.edu

[Hubertus Franke](https://orcid.org/0009-0005-0150-1055) IBM Research Yorktown Heights, USA frankeh@us.ibm.com

[Chloe Alverti](https://orcid.org/0000-0002-7965-0510) University of Illinois Urbana-Champaign Urbana, USA xalverti@illinois.edu

[Josep Torrellas](https://orcid.org/0000-0003-2595-5228) University of Illinois Urbana-Champaign Urbana, USA torrella@illinois.edu

## Abstract

The effectiveness of LLMs has triggered an exponential rise in their deployment, imposing substantial demands on inference clusters. Such clusters often handle numerous concurrent queries for different LLM downstream tasks. To handle multi-task settings with vast LLM parameter counts, Low-Rank Adaptation (LoRA) enables task-specific fine-tuning while sharing most of the base LLM model across tasks. Hence, it supports concurrent task serving with reduced memory requirements. However, existing designs face inefficiencies: they overlook workload heterogeneity, impose high CPU-GPU link bandwidth from frequent adapter loading, and suffer from head-of-line blocking in their schedulers.

To address these challenges, we present Chameleon, a novel LLM serving system optimized for many-adapter environments. Chameleon introduces two new ideas: adapter caching and adapteraware scheduling. First, Chameleon caches popular adapters in GPU memory, minimizing adapter loading times. For caching, it uses otherwise idle GPU memory, avoiding extra memory costs. Second, Chameleon uses a non-preemptive multi-queue scheduler to efficiently account for workload heterogeneity. In this way, Chameleon simultaneously prevents head of line blocking and starvation. Under high loads, Chameleon reduces the P99 and P50 TTFT latencies by 80.7% and 48.1%, respectively, over a state-of-the-art baseline, while improving the throughput by 1.5×.

