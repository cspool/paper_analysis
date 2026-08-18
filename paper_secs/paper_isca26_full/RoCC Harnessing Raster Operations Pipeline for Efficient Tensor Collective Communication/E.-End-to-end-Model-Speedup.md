# *E. End-to-end Model Speedup*

Simulating end-to-end LLM execution at cycle-level granularity would take *≈*300 days, so we used the networkcentric simulator Astra-Sim [49] following prior works [8], [45]. We generated Chakra [51] traces via an open-source tool [33], transformed the execution graph to enable finegrained overlap with RoCC. As shown in Figure 30(a), RoCC shows a 44% average speedup over the baseline by effectively hiding communication latency.

