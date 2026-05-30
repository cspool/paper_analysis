# D. Evaluate CPU Scalability

Having validated the effectiveness of the CPU, we further examine its scalability. In Figure 24, we assume that SLINFER initially has only two GPU nodes and zero CPU nodes, which are insufficient to handle all requests for 64 7B-sized models. As observed, continuously adding CPU nodes gradually increases the system's serving capacity to accommodate all requests. However, the scaling efficiency is lower compared to adding GPU nodes—roughly 3 to 4 CPU nodes are required to match the capacity of a single GPU node. This aligns with expectations, as CPUs have relatively lower compute power.

