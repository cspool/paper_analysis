# 3 Approach

## 3.1 Preliminary

GraphReader is built on a graph G = {V, E}, where each node v<sup>i</sup> ∈ V contains a key element k<sup>i</sup> and a set of summarized content, namely atomic facts A<sup>i</sup> . In other words, v<sup>i</sup> = {k<sup>i</sup> , Ai}. And

<span id="page-2-0"></span>> **[图片提取文字 (无描述)]:**
> Node (a) Graph Construction Key **Atomic Facts** Element Casa Loma Toronto **Atomic Facts** Danko Jones Atomic Facts From Chunk ID 1 p pl 1. "Never Too Loud" is the Gothic Danko Jones Neighbors fourth studio album Long Context Canadian hard rock band Extract Danko Jones. Canada Toronto Album Canada 2. Danko Jones is a Canadian Album hard rock trio from Toronto. (b) Graph Exploration Question Rational Plan Initial Notebook Never Too we need to identify the What is the name of the Loud performer band or castle in the city where None with "Never associated Album the performer of Never Too Loud", determine the Notebook Initialization Too Loud was formed? (2) Initial Node city where they were Selection formed, and then find out the name of any notable (1) Pre-planning read\_chunk(x,y,z) castle in that city. pop() Queue search\_more() 3 5 (3) Exploring Atomic Facts Queue read\_previous/ insert() read\_chunk(1,3,5) Queue Rational Plan Question 3 subsequent\_ 3 chunk() (4) Exploring Chunks Notebook \$ Chunk ID 1 termination() read\_neighbor\_node() (c) Answer Reasoning stop\_and\_read\_neighbor() **Updated Notebook** (5) Exploring Neighbors Response 1. The performer of Never Too Loud is Danko termination() Jones, which is a band from Toronto, Canada. Casa Loma 2. The text mentions that the castle in Toronto is Casa Loma.
![](_page_2_Figure_0.jpeg)

Figure 2: The illustration of our GraphReader approach, consisting of graph construction, graph exploration, and answer reasoning.

each edge  $e_{ij} \in \mathcal{E}$  represents the relationship between nodes  $v_i$  and  $v_j$ . This graph structure enables GraphReader to capture global information from the input document D within a limited context window, allowing it to decide whether to explore the current node in detail or jump to a neighboring node. During graph exploration, GraphReader collects supporting facts and terminates the exploration once sufficient information has been gathered to answer the question. As illustrated in Figure 2, the entire process of GraphReader consists of the following three phases: graph construction, graph exploration, and answer reasoning. The prompts utilized in these three stages are detailed in Appendix A, and a detailed example of our process can be found in Appendix I.

#### 3.2 Graph Construction

To extract nodes from a document D within the LLM's context limit, we first split D into chunks of maximum length L while preserving paragraph structure. For each chunk, we prompt the LLM to summarize it into atomic facts, the smallest indivisible facts that simplify the original text. We also

prompt the LLM to extract key elements from each atomic fact like essential nouns, verbs, and adjectives. After processing all chunks, we normalize the key elements as described by Lu et al. (2023) to handle lexical noise and granularity issues, creating a final set of key elements. We then construct each node  $v_i = (k_i, \mathcal{A}_i)$ , where  $k_i$  is a key element and  $\mathcal{A}_i$  is the set of atomic facts corresponding to  $k_i$ . Finally, we link two nodes  $v_i$  and  $v_j$  if key element  $k_i$  appears in  $\mathcal{A}_j$  and vice versa.

## 3.3 Graph Exploration

#### 3.3.1 Agent Initialization

Given a graph  $\mathcal{G}$  and a question Q, our goal is to design an agent that can autonomously explore the graph using predefined functions. The agent begins by maintaining a notebook to record supporting facts, which are eventually used to derive the final answer. Then the agent performs two key initializations: defining the rational plan and selecting the initial node.

**Rational Plan** To tackle complex real-world multi-hop questions, pre-planning the solution is

crucial. The agent breaks down the original question step-by-step, identifies the key information needed, and forms a rational plan.

Initial Node Choosing strategic starting points is essential for improving search efficiency. The agent evaluates the key elements of all nodes V and selects N initial nodes based on the question and the rational plan.

