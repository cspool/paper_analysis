# <span id="page-13-0"></span>I GraphReader Example

This section presents a case study of the GraphReader workflow. Figure [20](#page-24-1) displays the posed question alongside the answer and pertinent supporting passages. Subsequently, Figure [21](#page-25-0) delineates the methodology for constructing the graph. Figure [22](#page-26-0) further elaborates on the initialization of a pre-planned rational path by GraphReader and the selection of initial nodes. Figure [23](#page-27-0) illustrates the sequence of function invocations during the exploration phase. Finally, Figure [24](#page-28-0) showcases how GraphReader formulates the answer by leveraging the insights obtained through exploration.

<span id="page-14-0"></span>

| Dataset                                                                                                                                                                                                                                                                                                                            | #Avg. Function Call | Stage               | Stage Ratio(%) | Function               | Call Ratio(%) |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|---------------------|----------------|------------------------|---------------|
|                                                                                                                                                                                                                                                                                                                                    |                     |                     | 42.0           | read_chunk             | 46.5          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | stop_and_read_neighbor | 53.5          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | search_more            | 12.1          |
| HotpotQA                                                                                                                                                                                                                                                                                                                           |                     |                     | 31.9           | read_previous_chunk    | 21.1          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | read_subsequent_chunk  | 22.9          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | termination            | 43.9          |
| Exploring Atomic Facts<br>3.0<br>Exploring Chunks<br>Exploring Neighbors<br>Exploring Atomic Facts<br>3.2<br>2WikiMultihopQA<br>Exploring Chunks<br>Exploring Neighbors<br>Exploring Atomic Facts<br>3.5<br>MuSiQue<br>Exploring Chunks<br>Exploring Neighbors<br>Exploring Atomic Facts<br>3.9<br>NarrativeQA<br>Exploring Chunks | 26.1                | read_neighbor_node  | 35.5           |                        |               |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | termination            | 65.5          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     | 40.4           | read_chunk             | 48.6          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | stop_and_read_neighbor | 51.4          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | search_more            | 14.5          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     | 34.5           | read_previous_chunk    | 25.1          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | read_subsequent_chunk  | 23.3          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | termination            | 37.1          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     | 25.1           | read_neighbor_node     | 37.3          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | termination            | 62.7          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     | 40.0           | read_chunk             | 41.3          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | stop_and_read_neighbor | 58.7          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | search_more            | 19.1          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     | 31.2           | read_previous_chunk    | 26.6          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | read_subsequent_chunk  | 25.7          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | termination            | 28.6          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     | 28.8           | read_neighbor_node     | 40.1          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | termination            | 59.9          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     | 32.5           | read_chunk             | 64.5          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | stop_and_read_neighbor | 35.5          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | search_more            | 4.1           |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     | 54.3           | read_previous_chunk    | 35.3          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | read_subsequent_chunk  | 32.6          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | termination            | 28.0          |
|                                                                                                                                                                                                                                                                                                                                    |                     | Exploring Neighbors | 13.2           | read_neighbor_node     | 51.4          |
|                                                                                                                                                                                                                                                                                                                                    |                     |                     |                | termination            | 48.6          |

Table 8: Statistics of function calls on MuSiQue and NarrativeQA.

<span id="page-14-1"></span>

|                    |      | Sample Dimension |          |        | Sample & Node Dimension |                   |           |                  |      |  |
|--------------------|------|------------------|----------|--------|-------------------------|-------------------|-----------|------------------|------|--|
| dataset            |      |                  | node num |        | atomic facts num        | neighbor node num |           | atomic facts num |      |  |
|                    | avg. | max              | avg.     | max    | avg. avg.               | avg. max          | avg. avg. | avg. max         |      |  |
| HotpotQA           |      | 583.8            | 1945.0   | 244.0  | 645.0                   | 10.1              | 263.1     | 2.1              | 17.8 |  |
| 2WikiMultihopQA    |      | 515.8            | 1691.0   | 217.7  | 545.0                   | 9.2               | 215.7     | 2.1              | 17.0 |  |
| MusiQue            |      | 1029.4           | 2142.0   | 419.9  | 586.0                   | 9.3               | 253.4     | 2.1              | 15.6 |  |
| NarrativeQA        |      | 966.0            | 3110.0   | 515.5  | 1296.0                  | 10.3              | 652.6     | 2.3              | 50.0 |  |
|                    | 16k  | 1741.6           | 3822.0   | 749.7  | 1043.0                  | 9.4               | 231.0     | 2.2              | 17.1 |  |
|                    | 32k  | 2827.3           | 5086.0   | 1257.4 | 1694.0                  | 9.8               | 263.3     | 2.2              | 29.3 |  |
| HotpotWikiQA-mixup | 64k  | 5054.1           | 8918.0   | 2360.0 | 3015.0                  | 10.4              | 227.2     | 2.3              | 17.1 |  |
|                    | 128k | 8828.5           | 14592.0  | 4437.9 | 5182.0                  | 11.1              | 302.0     | 2.4              | 19.2 |  |
|                    | 256k | 14853.3          | 24981.0  | 8632.8 | 9478.0                  | 12.2              | 427.6     | 2.5              | 27.8 |  |

Table 9: Graph statistical data. Under the Sample dimension, "avg." indicates the average number of nodes in each graph, and "max" refers to the largest node count across all graphs. The same logic applies to atomic facts num. In the Sample & Node dimensions, "avg. avg." denotes the average of the average neighbor node counts per graph, and "avg. max" means the average of the maximum neighbor node counts per graph. This approach is also used for counting atomic facts num.

<span id="page-15-0"></span>You are now an intelligent assistant tasked with meticulously extracting both key elements and atomic facts from a long text.

- 1. Key Elements: The essential nouns (e.g., characters, times, events, places, numbers), verbs (e.g., actions), and adjectives (e.g., states, feelings) that are pivotal to the text's narrative.
- 2. Atomic Facts: The smallest, indivisible facts, presented as concise sentences. These include propositions, theories, existences, concepts, and implicit elements like logic, causality, event sequences, interpersonal relationships, timelines, etc.

## Requirements:

## #####

- 1. Ensure that all identified key elements are reflected within the corresponding atomic facts.
- 2. You should extract key elements and atomic facts comprehensively, especially those that are important and potentially query-worthy and do not leave out details.
- 3. Whenever applicable, replace pronouns with their specific noun counterparts (e.g., change I, He, She to actual names).
- 4. Ensure that the key elements and atomic facts you extract are presented in the same language as the original text (e.g., English or Chinese).
- 5. You should output a total of key elements and atomic facts that do not exceed 1024 tokens.
- 6. Your answer format for each line should be: [Serial Number], [Atomic Facts], [List of Key Elements, separated with '|']

#####

## Example:

#####

User:

One day, a father and his little son ......

## Assistant:

1. One day, a father and his little son were going home. | father | little son | going home

2. ......

#####

Figure 6: The prompt for key elements and atomic facts extraction.

<span id="page-16-0"></span>As an intelligent assistant, your primary objective is to answer the question by gathering supporting facts from a given article. To facilitate this objective, the first step is to make a rational plan based on the question. This plan should outline the step-by-step process to resolve the question and specify the key information required to formulate a comprehensive answer.

Example:

#####

User: Who had a longer tennis career, Danny or Alice?

Assistant: In order to answer this question, we first need to find the length of Danny's and Alice's tennis careers, such as the start and retirement of their careers, and then compare the two.

#####

Figure 7: The prompt for rational plan.

As an intelligent assistant, your primary objective is to answer questions based on information contained within a text. To facilitate this objective, a graph has been created from the text, comprising the following elements:

- 1. Text Chunks: Chunks of the original text.
- 2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
- 3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic facts derived from different text chunks.

Your current task is to check a list of nodes, with the objective of selecting the most relevant initial nodes from the graph to efficiently answer the question. You are given the question, the rational plan, and a list of node key elements. These initial nodes are crucial because they are the starting point for searching for relevant information.

## Requirements:

## #####

- 1. Once you have selected a starting node, assess its relevance to the potential answer by assigning a score between 0 and 100. A score of 100 implies a high likelihood of relevance to the answer, whereas a score of 0 suggests minimal relevance.
- 2. Present each chosen starting node in a separate line, accompanied by its relevance score. Format each line as follows: Node: [Key Element of Node], Score: [Relevance Score].
- 3. Please select at least 10 starting nodes, ensuring they are non-repetitive and diverse.
- 4. In the user's input, each line constitutes a node. When selecting the starting node, please make your choice from those provided, and refrain from fabricating your own. The nodes you output must correspond exactly to the nodes given by the user, with identical wording.

#####

Example: ##### User:

Question: {QUESTION} Plan: {RATIONAL PLAN}

Nodes: {LIST OF KEY ELEMENTS}

Assistant:{LIST OF SELECTED NODES}

#####

Finally, I emphasize again that you need to select the starting node from the given Nodes, and it must be consistent with the words of the node you selected. Please strictly follow the above format. Let's begin.

Figure 8: The prompt for initial node selection.

As an intelligent assistant, your primary objective is to answer questions based on information contained within a text. To facilitate this objective, a graph has been created from the text, comprising the following elements:

- 1. Text Chunks: Chunks of the original text.
- 2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
- 3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic facts derived from different text chunks.

Your current task is to check a node and its associated atomic facts, with the objective of determining whether to proceed with reviewing the text chunk corresponding to these atomic facts. Given the question, the rational plan, previous actions, notebook content, and the current node's atomic facts and their corresponding chunk IDs, you have the following Action Options: #####

- 1. read\_chunk(List[ID]): Choose this action if you believe that a text chunk linked to an atomic fact may hold the necessary information to answer the question. This will allow you to access more complete and detailed information.
- 2. stop\_and\_read\_neighbor(): Choose this action if you ascertain that all text chunks lack valuable information.

#####

## Strategy:

#####

- 1. Reflect on previous actions and prevent redundant revisiting nodes or chunks.
- 2. You can choose to read multiple text chunks at the same time.
- 3. Atomic facts only cover part of the information in the text chunk, so even if you feel that the atomic facts are slightly relevant to the question, please try to read the text chunk to get more complete information.

#####

