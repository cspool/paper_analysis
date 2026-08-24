# 5 Evaluation

#### 5.1 Methodology

Testbed. We evaluate Strata and baselines on two platforms. The H200 platform is a node equipped with 8 NVIDIA H200 GPUs interconnected with NVLink, an Intel Sapphire Rapids CPU, and 1.6TB of DRAM. Each GPU is connected to the CPU

<span id="page-6-0"></span>

|            | LooGLE | NarrativeQA | ReviewMT | ShareGPT |
|------------|--------|-------------|----------|----------|
| avg. in    | 21613  | 54797       | 17708    | 680.9    |
| avg. out   | 15.60  | 13.00       | 208.3    | 260.9    |
| # contexts | 105    | 50          | 100      | -        |
| # queries  | 2410   | 1461        | 1092     | 200869   |

Table 1. Dataset statistics.

via a PCIe 5.0 x16 link, offering up to 64 GB/s of peak bandwidth (unidirectional). The GH200 platform is a GH200 Grace Hopper superchip [\[32\]](#page-12-20) node, which contains one NVIDIA H100 GPU integrated with one NVIDIA Grace 64-core ARM CPU. The GH200 system is equipped with 464GB of LPDDR5X DRAM, providing up to 384 GB/s of memory bandwidth (unidirectional) to the CPU.

Baselines. We compare Strata with following state-of-theart baselines.

vLLM [\[22\]](#page-11-9) is a popular open-source serving engine. Additionally, vLLM-LMCache enables hierarchical caching on vLLM using the official community extension of LMCache [\[25\]](#page-12-8). For our benchmarks, we used vLLM v0.8.5 and LMCache v0.2.1. The LMCache chunk size is set to 256 as default and vLLM page size was set to 32 in line with prior work [\[11\]](#page-11-6).

TensorRT-LLM [\[33\]](#page-12-7) is an open-source serving library from NVIDIA, specialized for NVIDIA GPUs. Additionally, TensorRT-HiCache enables hierarchical caching on top of TensorRT-LLM through its automatic CPU memory offloading feature. We used TensorRT-LLM v0.17.0 in our benchmarks, with the page size also set to 32 as default.

SGLang [\[45\]](#page-12-4) is an open-source serving engine that delivers comparable performance to vLLM, while offering a more lightweight and customizable architecture. To enable a direct comparison to Strata, we implemented SGLang-HiCache which incorporates a state-of-the-art layer-wise KV cache transfer overlapping and hierarchical caching implementation using cudaMemcpyAsync transfers, which is in line with prior work including CachedAttention [\[11\]](#page-11-6), Pensieve [\[44\]](#page-12-5) and FlashGen [\[18\]](#page-11-13). We used SGLang v0.4.5 for all three systems. We set the page size for SGLang and Strata to 1 (SGLang's default), and the page size for SGLang-HiCache to 32 to be consistent with other hierarchical cache baselines. Models. We utilize three popular open-source LLMs with long context capabilities, spanning small, medium, and large sizes: Llama-3.1-8B-Instruct [\[28\]](#page-12-1) (128k context window), Qwen2.5-14B-Instruct-1M [\[39\]](#page-12-0) (1M context window), and Llama-3.1-70B-Instruct [\[28\]](#page-12-1) (128k context window). We served the 8B and 14B models using a single GPU, and served the 70B model using 4 GPUs configured with tensor parallelism.

Datasets. We construct workloads from three long context datasets. LooGLE [\[23\]](#page-11-8) features long documents from diverse sources such as arXiv, Wikipedia, and movie/TV scripts. In our benchmarks, we use its Wikipedia portion, which provides both long and short queries paired with the documents. NarrativeOA [21] is an influential long-context dataset for testing models' reading comprehension capabilities, featuring even longer context examples than LooGLE. We filtered documents exceeding 128k tokens because of context window limit of the test models, and sampled 50 documents from the remainder. These two datasets mirror classic RAG use cases, in which extensive contexts are repeatedly queried by multiple users over time like question-answering systems over technical manuals [12]. ReviewMT [38] is a multi-agent conversation dataset, where agents simulate reviewers to converse about the quality of technical papers to make final decisions. This represents a typical agentic workflow involving long contexts. We also include a dataset to evaluate Strata's performance in short-context scenarios. *ShareGPT* is a popular conversational dataset comprised of a large collection of conversation histories from thousands of users, and was used in prior hierarchical KV caching studies [11, 44]. Table 1 summarizes the characteristics of these datasets.

Since individual query timestamps are not available in these datasets, we simulate query arrivals using a Poisson distribution to benchmark the system using varying request rates, following prior works [18, 44]. For conversational benchmarks (ReviewMT and ShareGPT), we preserve dependencies across conversation rounds. Consistent with the methodology in Pensieve [44], we insert a 60-second "thinking time" between an LLM's response and the user's subsequent query for ShareGPT. For the long-context benchmarks, queries are randomly sampled from the dataset. In §5.3.3, we further examine the performance characteristics under different workload patterns. To avoid execution timeouts, we cap the maximum number of in-flight queries at 128 across all benchmarks. GPU memory is allocated according to each serving engine's default policy to ensure fairness and performance. An exception is the ShareGPT dataset, where we restrict GPU memory to approximately 500K tokens to highlight the behavior of hierarchical caching baselines. For caching configurations that utilize CPU memory, we allocate 1 TB of system DRAM as pinned memory (400 GB on GH200 due to platform limits). Disk storage is not used in all benchmarks due to limited support in baseline systems.

#### 5.2 End-to-end Performance Comparison

Strata is designed to improve long-context serving by reducing response latency and increasing overall throughput. Accordingly, we evaluate the system using two primary metrics: average Time To First Token (TTFT) and output token throughput. TTFT captures query response time, a key determinant of user experience, while output token throughput is a widely adopted metric for characterizing LLM serving system performance [1].

