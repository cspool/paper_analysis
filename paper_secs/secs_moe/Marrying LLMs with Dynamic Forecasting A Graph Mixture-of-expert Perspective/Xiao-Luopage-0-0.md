# Xiao Luo[\\*](#page-0-0)

University of California, Los Angeles xiaoluo@cs.ucla.edu

### Abstract

Dynamical system modeling is a crucial area of research in machine learning with extensive applications in physics and social science. Recent data-driven approaches often employ graph neural networks (GNNs) to learn relationships in dynamical systems using message passing mechanisms. Despite their advancements, these methods often suffer from performance degradation when it comes to potential environmental change with distribution shifts in real-world applications. In this work, we propose a new perspective which leverages large language models (LLMs) to enhance the generalization capabilities of dynamical system modeling. In particular, we develop a novel framework named LLM Judge with Graph Mixtureof-expert (LEGO), which incorporates multiple graph experts to learn diverse dynamics within the systems. More importantly, LEGO utilizes LLMs with hierarchical prompts at object, edge, and system levels as a context-aware routing function to determine which experts carry the most relevant information to different environments. The whole framework is optimized by updating the weights and expert parameters in an alternative fashion. Extensive experiments across various datasets demonstrate the effectiveness of our proposed LEGO in comparison to extensive baselines.

