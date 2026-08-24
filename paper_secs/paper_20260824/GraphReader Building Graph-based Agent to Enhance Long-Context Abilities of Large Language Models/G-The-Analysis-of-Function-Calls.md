# **G** The Analysis of Function Calls

To verify the rationality and utility of agent actions under various circumstances of GraphReader, we made statistics on its function calls at each stage across two datasets. From the statistical results in Table 8, it can be observed that each piece of data

will perform an average of 3 to 4 actions, corresponding to the average number of function calls in the table. This indicates the effectiveness of the graph we constructed, with GraphReader being able to swiftly locate key information while minimizing resource usage. Furthermore, each action has a certain probability of being chosen, justifying the rationality of the action set. Among them, the most commonly used action on multi-hop QA tasks is to read neighbor nodes, and the most common action on single-hop QA tasks is to read chunks. This difference is caused by the fact that multi-hop questions need to gather information contained by multiple nodes to answer questions, while singlehop data sets often require only one atomic fact.

