# *A. Design Space*

QuCo addresses the combinatorial complexity of ATTenabled kernels, where selecting tile sizes and queue slot counts across multiple operand queues leads to a vast design space. The number of valid configurations grows exponentially with the number of queues and tuning parameters. Table I summarize the possible configurations for each workload. For our evaluation, we constrain the search space to practical ranges: tile sizes from 64 to 8,192 elements7, and queue slots from 1 to 8. This results in billions to quadrillions of possible combinations. Manually exploring this space would be prohibitively expensive, but QuCo simplifies the process by automatically identifying high-performing configurations in a single pass, eliminating the need for manual tuning.

