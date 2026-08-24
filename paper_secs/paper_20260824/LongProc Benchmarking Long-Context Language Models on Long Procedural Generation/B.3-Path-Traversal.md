# **B.3 Path Traversal**

This tasks requires LCLMs to keep track of a route between two cities in a hypothetical transit network between cities. Please refer to Prompt [H.3](#page-44-0) for a concrete data example and the detailed prompt used for evaluation.

Essentially, the underlying problem is to traverse a path between two nodes (cities) in a directly acyclic graph. Each data instance contains a graph *G* = ⟨*V*, *E*⟩, where *V* represents the set of nodes and *E* represents the set of directed edges. The task asks LLMs to find the path between a source node *v<sup>s</sup>* and a destination node *v<sup>d</sup>* .

We construct the graph and source-target node pairs with two important constraints: 1) there exists exactly one path from the source node *v<sup>s</sup>* to the target node *v<sup>d</sup>* , and 2) each node *v<sup>i</sup>* along the path has exactly one outgoing edge to the next node *v<sup>j</sup>* along the path. This design ensures that LLMs can find the path by simply following the next node from the initial node, without requiring any search algorithms. To cast the problem into a natural language task, we assign each node a real city name from a pre-collected set and describe each edge using natural language. The edge descriptions follow this template: "[city a] is a lively city. You can travel from [city a] to [city b] by [transit]." The complete list of edges is provided to LLMs as a natural language description of the graph. We construct the data sets for the three difficulty levels by varying the number of the cities in the output path.

