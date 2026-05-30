# VIII. CONCLUSION

This paper presents VECTORLITERAG, a latency-aware orchestration framework for Retrieval-Augmented Generation (RAG) systems that explicitly manages the tight coupling between vector retrieval and LLM inference. We show that under skewed access patterns, variability in retrieval latency interacts with inference batching, causing tail effects amplification that cannot be mitigated by optimizing either stage in isolation.

VECTORLITERAG is driven by the insight that meeting strict RAG SLOs requires balancing batching behavior rather than maximizing instantaneous GPU utilization. By coordinating retrieval progress with inference scheduling, VECTORLITERAG suppresses tail cascades and sustains predictable end-to-end latency under bursty workloads. This enables SLO compliance across a substantially wider operating regime, supporting up to 1.5× higher request rates than baseline RAG systems. Across extensive evaluation, we demonstrate that these benefits generalize across latency targets, hardware configurations, and LLM input/output lengths. VECTORLITERAG further exposes explicit control knobs that allow RAG operators to trade throughput for tail latency under constrained GPU memory budgets, making it practical for real-world deployment.

#### IX. ACKNOWLEDGMENT

This research was supported in part through cyber-infrastructure research resources and services provided by the Partnership for an Advanced Computing Environment (PACE) at the Georgia Institute of Technology, Atlanta, Georgia, USA. This work was partially supported by gifts from Google and AMD. The views and conclusions contained herein are those of the authors and should not be interpreted as representing the official policies or endorsements, either expressed or implied, of Georgia Tech.

#### ARTIFACT APPENDIX

#### *A. Abstract*

The artifact includes the complete source code of the coreVECTORLITERAG system, together with our modified FAISS library used for hybrid CPU–GPU vector search. To ensure reproducibility of both preprocessing and evaluation, we also provide a collection of shell scripts and Python utilities that automate the full experimental workflow, including dataset preparation, index construction, performance profiling, and end-to-end RAG pipeline evaluation. These scripts are designed to reproduce all major results reported in the paper.

All code and supporting materials are publicly available on GitHub https://github.com/sitar-lab/VectorLiteRAG-AE and Zenodo https://zenodo.org/records/18195323

## *B. Artifact Check-list*

- Program: Modified FAISS library, vLLM
- Compilation: gcc-11.3, nvcc-12.1, cmake.
- Models: Llama-3 8B, Llama-3 70B, and Qwen-3 32B.
- Datasets: MS ORCAS and NVIDIA Wiki-All.
- Run-time Environment: RHEL 9 with Anaconda3.
- Hardware: Single node equipped with 8 NVIDIA L40S GPUs and 8 NVIDIA H100 GPUs.
- Metrics: SLO attainment, end-to-end latency, vector search hit rate estimation.
- Output: CSV logs and visualization plots.
- Disk Space Required: ∼256 GB for evaluation; ∼1.5 TB for preprocessing and index construction.
- Workflow Preparation Time: 40–50 hours.
- Experiment Completion Time: 10 hours.
- Publicly Available?: Yes.
- Code Licenses?: CC BY 4.0
- Archived(DOI)?: https://zenodo.org/records/18195323

### *C. Description*

- *1) How to access:* All source cod and scripts are accessible via github repository.
- *2) Hardware dependencies:* All experiments were conducted on a single L40S node or a single H100 node, each equipped with 8 GPUs. Because larger language models rely on tensor model parallelism, the H100 system is expected to provide NVLink connectivity to ensure reproducible performance. The L40S node was configured with a 32-core Intel CPU, and the H100 node with a 64-core Intel CPU. CPU core count is an important factor, as a substantial portion of the workload executes on the host processor.
- *3) Software dependencies:* The evaluation environment was run on RHEL 9 (or a compatible Linux distribution) using Anaconda3. Successful compilation of the FAISS library depends on specific toolchain versions, including Python 3.10, GCC 11.3, and NVCC 12.1. Intel MKL is also required to support vectorized CPU operations.
- *4) Datasets:* The Wikiall benchmark is directly downloadable. The ORCAS 1K and ORCAS 2K benchmarks require both the MS ORCAS dataset and the English Wikipedia dump, which are publicly accessible but require long preprocessing.

#### *D. Installation and Testing*

## *1) Installation:*

```
# Create conda environment
cd VectorLiteRAG
conda create -n vlite -f ./scripts/env.yml
conda activate vlite
# Build faiss library
```

./scripts/build.sh

### *2) Preprocessing:*

