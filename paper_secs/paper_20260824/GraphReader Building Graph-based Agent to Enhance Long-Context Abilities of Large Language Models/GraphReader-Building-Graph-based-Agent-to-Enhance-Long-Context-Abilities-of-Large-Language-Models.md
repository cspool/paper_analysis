# GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models

Shilong Li\*1, Yancheng He\*1, Hangyu Guo\*1, Xingyuan Bu\*†‡1, Ge Bai¹, Jie Liu²,³, Jiaheng Liu¹, Xingwei Qu⁴, Yangguang Li³, Wanli Ouyang²,³, Wenbo Su¹, Bo Zheng¹

Alibaba Group <sup>2</sup>The Chinese University of Hong Kong

Shanghai AI Laboratory <sup>4</sup>University of Manchester

zhuli.lsl@taobao.com, xingyuanbu@gmail.com

#### **Abstract**

Long-context capabilities are essential for large language models (LLMs) to tackle complex and long-input tasks. Despite numerous efforts made to optimize LLMs for long contexts, challenges persist in robustly processing long inputs. In this paper, we introduce GraphReader, a graph-based agent system designed to handle long texts by structuring them into a graph and employing an agent to explore this graph autonomously. Upon receiving a question, the agent first undertakes a step-by-step analysis and devises a rational plan. It then invokes a set of predefined functions to read node content and neighbors, facilitating a coarse-to-fine exploration of the graph. Throughout the exploration, the agent continuously records new insights and reflects on current circumstances to optimize the process until it has gathered sufficient information to generate an answer. Experimental results on the LV-Eval dataset reveal that GraphReader, using a 4k context window, consistently outperforms GPT-4-128k across context lengths from 16k to 256k by a large margin. Additionally, our approach demonstrates superior performance on four challenging single-hop and multi-hop benchmarks.

## 1 Introduction

Large language models (LLMs) have made great progress on natural language understanding and generation (Zhao et al., 2023; Liu et al., 2024a; Feng et al., 2022; Peng et al., 2020; Xv et al., 2022; Peng et al., 2023b; Bu et al., 2021). However, transformer-based LLMs still struggle in handling long contexts due to the limitation of context window and memory usage.

Current techniques for solving the long-context tasks of LLMs can be divided into two perspectives: 1) Model-level, which includes finetuning with modified positional embeddings (Chen et al.,

<span id="page-0-0"></span>> **[图片提取文字 (无描述)]:**
> 35 30 Average Scores (%) 20 25 20 21 21 10 5 16k 32k 64k 128k 256k Input Length GraphReader GPT-4-128k (chunk) ReadAgent Ada-002 (top-1) GPT-4-128k BM25 (top-1) GPT-4-128k (chunk w/ notes)
![](_page_0_Figure_10.jpeg)

Figure 1: Performance on LV-Eval at 5 context length levels. GraphReader outperforms existing open-sourced and closed-source models while demonstrating a scalable performance in very long contexts. In contrast, other models exhibit a significant decrease in performance as context length increases.

2023b; Zhu et al., 2023; Peng et al., 2023a; Ding et al., 2024), and applying transformer variants with modified attention mechanisms (Dai et al., 2019; Munkhdalai et al., 2024; Gu and Dao, 2023); 2) Agent-level, *i.e.*, employing retrieval-augmented LLM or agent to process long contexts with a limited context window LLM (Nakano et al., 2021; Lee et al., 2024).

However, model-level methods typically train LLMs with target length texts, posing challenges in constructing training datasets and incurring high training costs (Zhu et al., 2023). Additionally, long-context LLMs optimized with these methods tend to overlook crucial details in long contexts, known as "lost in the middle" (Liu et al., 2024b), limiting their ability to address complex tasks, such as multi-hop questions. Agent-level approaches transform input text into a tree (Chen et al., 2023a) or paginated pages (Lee et al., 2024), failing to capture multi-hop and long-range dependencies, thus

<sup>\*</sup> First four authors contributed equally.

<sup>†</sup> Corresponding Author. ‡ Project Leader.

limiting their effectiveness on very long contexts, as shown in Figure [1.](#page-0-0)

To address these issues, we propose a graphbased agent named GraphReader. As illustrated in Figure [2,](#page-2-0) GraphReader first segments long texts into discrete chunks, extracts essential information, and compresses these into key elements and atomic facts. These key elements and facts are then used to construct a graph with nodes representing key elements and their associated atomic facts. This graph structure effectively captures long-range dependencies and multi-hop relationships within long text. Subsequently, GraphReader autonomously explores this graph using predefined functions, guided by a step-by-step rational plan. Based on a given question, the agent progressively accesses information from coarse key elements and atomic facts to detailed original text chunks, taking notes and reflecting until it gathers sufficient information to generate an answer. In summary, our main contributions are threefold:

- We introduce GraphReader, a novel agent system designed to organize long texts into a graph structure, leveraging predefined functions and notebook to facilitate planning and reflection during exploration.
- GraphReader establishes a scalable long-context capability based on a 4k context window, demonstrating performance that is comparable to or surpasses GPT-4 with a 128k context window across varying context lengths.
- Extensive experiments conducted on four challenging benchmarks demonstrate that GraphReader achieves superior performance in complex single-hop and multi-hop QA tasks.

