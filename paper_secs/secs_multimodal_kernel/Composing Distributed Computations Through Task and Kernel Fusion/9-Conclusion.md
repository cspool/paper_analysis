# 9 Conclusion

We introduced Diffuse, a system that performs task and kernel fusion on streams of distributed tasks, enabling optimizations that improve data reuse and remove allocations of distributed data structures in end user programs. Diffuse leverages a scale-free intermediate representation of distributed computation and data to perform these analyses in a scalable manner. These techniques enable Diffuse to compose computations in and across cuPyNumeric and Legate Sparse, matching or exceeding the performance of hand-tuned code.

