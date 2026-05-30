# Abstract

LLM serving providers typically offer a suite of structurally similar models, known as model families, such as the opensource Llama2 series featuring 7B, 13B, and 70B models. While numerous optimizations for LLM serving have been proposed, the potential for leveraging synergies between models within the same family has not been thoroughly explored. This paper introduces MFS, an innovative multitiered LLM model family serving system to exploit the structural similarities and parameter redundancies across different scales of models within a family. By utilizing a novel fine-tuning technique called Knowledge Precipitation, MFS restructures the largest model in a family to encapsulate smaller models within its architecture, enabling a unified multi-tiered serving pipeline. Based on the multi-tiered model, MFS realizes a highly parallelized tiered-level batching approach, significantly enhancing system efficiency. It also enables the sharing of intermediate features and KV-cache between models and facilitates multi-level sampling techniques during the inference phase. Experimental results demonstrate that MFS achieves substantial improvements over existing methods, including a 56.1% reduction in end-to-end token generation latency and a 47.8% decrease in GPU memory footprint without compromising the quality of generated content.

![](_page_0_Figure_21.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 Interna](https://creativecommons.org/licenses/by/4.0)[tional License.](https://creativecommons.org/licenses/by/4.0)

EUROSYS '26, Edinburgh, Scotland Uk © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2212-7/26/04 <https://doi.org/10.1145/3767295.3769355>

CCS Concepts: • Computing methodologies → Machine learning;• Computer systems organization→Distributed architectures.

Keywords: Large Language Models, Model Serving, Distributed Systems, Inference Optimization

#### ACM Reference Format:

Yunxuan Zhang, Hao Wang, Han Tian, Liu Yang, Xudong Liao, Wenxue Li, Ping Yin, Bowen Liu, and Kai Chen. 2026. MFS: An Efficient Model Family Serving System for LLMs. In European Conference on Computer Systems (EUROSYS '26), April 27–30, 2026, Edinburgh, Scotland Uk. ACM, New York, NY, USA, [15](#page-14-0) pages. [https:](https://doi.org/10.1145/3767295.3769355) [//doi.org/10.1145/3767295.3769355](https://doi.org/10.1145/3767295.3769355)

