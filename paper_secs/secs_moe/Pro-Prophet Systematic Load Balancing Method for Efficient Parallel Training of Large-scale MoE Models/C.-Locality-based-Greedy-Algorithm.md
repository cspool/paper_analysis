# *C. Locality-based Greedy Algorithm*

The performance model can accurately estimate the execution time of a MoE layer deploying any expert placements. However, it is necessary to determine a communicationefficient one in various load imbalance scenarios. There are 2 <sup>N</sup>∗<sup>E</sup> potential lightweight expert placements. The brute force search algorithm is time-consuming and could be a performance bottleneck.

Therefore, the planner offers an efficient greedy search algorithm shown in Algorithm [1.](#page-5-0) Taking the results of gate network gating, s and n as input, Algorithm [1](#page-5-0) iteratively generates and evaluates for a better expert placement until the load is balanced. Finally, it outputs a communication-efficient expert placement P oE.

Initially, the algorithm estimates the execution time of a MoE layer without implementing any lightweight expert placements and records it as minimum time. Then it employs two greedy strategies to generate a lightweight expert placement that optimizes the load of devices. Specifically, it prioritizes the expert with the higher number of responsible inputs for selection and transfers its parameters to devices that hold more inputs processed by the expert. The algorithm maintains a list of L and n bottoms to record the expert placement. Then the algorithm evaluates the expert placement using the performance model. It updates the minimum time and a counter if the current expert placement achieves a better performance. The search process is repeated until the load is imbalanced. The condition of the balanced load is

$$\max(H) - \min(H) < \alpha \frac{I}{E},\tag{7}$$

where I is the number of inputs training in an iteration and α is a regulable coefficient for different requirements of load balance.

As the search algorithm is required to run during the MoE model training, we define a primitive Plan to describe this search process. As mentioned in Sec. [II,](#page-1-0) the input distributions of adjacent iterations are similar, which inspired us to predict the distribution and reduce the frequency of execution of the algorithm. Based on the inspiration, the planner upgrades the algorithm to a locality-based one. Users can adjust the frequency of the search algorithm flexibly for better training efficiency.

