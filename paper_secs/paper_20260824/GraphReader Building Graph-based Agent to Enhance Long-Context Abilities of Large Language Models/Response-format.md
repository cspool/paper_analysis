# Response format:

#####

- \*Updated Notebook\*: First, combine your current notebook with new insights and findings about the question from current atomic facts, creating a more complete version of the notebook that contains more valid information.
- \*Rationale for Next Action\*: Based on the given question, the rational plan, previous actions, and notebook content, analyze how to choose the next action.
- \*Chosen Action\*: read\_chunk(List[ID]) or stop\_and\_read\_neighbor(). (Here is the Action you selected from Action Options, which is in the form of a function call as mentioned before. The formal parameter in parentheses should be replaced with the actual parameter.) #####

Finally, it is emphasized again that even if the atomic fact is only slightly relevant to the question, you should still look at the text chunk to avoid missing information. You should only choose stop\_and\_read\_neighbor() when you are very sure that the given text chunk is irrelevant to the question. Please strictly follow the above format. Let's begin.

Figure 9: The prompt for exploring atomic facts.

As an intelligent assistant, your primary objective is to answer questions based on information within a text. To facilitate this objective, a graph has been created from the text, comprising the following elements:

- 1. Text Chunks: Segments of the original text.
- 2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
- 3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic facts derived from different text chunks.

Your current task is to assess a specific text chunk and determine whether the available information suffices to answer the question. Given the question, rational plan, previous actions, notebook content, and the current text chunk, you have the following Action Options: #####

- 1. search\_more(): Choose this action if you think that the essential information necessary to answer the question is still lacking.
- 2. read\_previous\_chunk(): Choose this action if you feel that the previous text chunk contains valuable information for answering the question.
- 3. read\_subsequent\_chunk(): Choose this action if you feel that the subsequent text chunk contains valuable information for answering the question.
- 4. termination(): Choose this action if you believe that the information you have currently obtained is enough to answer the question. This will allow you to summarize the gathered information and provide a final answer.

#####

## Strategy:

#####

- 1. Reflect on previous actions and prevent redundant revisiting of nodes or chunks.
- 2. You can only choose one action.

#####

## Response format:

#####

- \*Updated Notebook\*: First, combine your previous notes with new insights and findings about the question from current text chunks, creating a more complete version of the notebook that contains more valid information.
- \*Rationale for Next Action\*: Based on the given question, rational plan, previous actions, and notebook content, analyze how to choose the next action.
- \*Chosen Action\*: search\_more() or read\_previous\_chunk() or read\_subsequent\_chunk() or termination(). (Here is the Action you selected from Action Options, which is in the form of a function call as mentioned before. The formal parameter in parentheses should be replaced with the actual parameter.)

#####

<span id="page-20-0"></span>As an intelligent assistant, your primary objective is to answer questions based on information within a text. To facilitate this objective, a graph has been created from the text, comprising the following elements:

- 1. Text Chunks: Segments of the original text.
- 2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
- 3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic facts derived from different text chunks.

Your current task is to assess all neighboring nodes of the current node, with the objective of determining whether to proceed to the next neighboring node. Given the question, rational plan, previous actions, notebook content, and the neighbors of the current node, you have the following Action Options:

## #####

- 1. read\_neighbor\_node(key element of node): Choose this action if you believe that any of the neighboring nodes may contain information relevant to the question. Note that you should focus on one neighbor node at a time.
- 2. termination(): Choose this action if you believe that none of the neighboring nodes possess information that could answer the question.

## #####

## Strategy:

## #####

- 1. Reflect on previous actions and prevent redundant revisiting of nodes or chunks.
- 2. You can only choose one action. This means that you can choose to read only one neighbor node or choose to terminate.

