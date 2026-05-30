# Abstract

We introduce Diffuse, a system that dynamically performs task and kernel fusion in distributed, task-based runtime systems. The key component of Diffuse is an intermediate representation of distributed computation that enables the necessary analyses for the fusion of distributed tasks to be performed in a scalable manner. We pair task fusion with a JIT compiler to fuse together the kernels within fused tasks. We show empirically that Diffuse's intermediate representation is general enough to be a target for two real-world, task-based libraries (cuPyNumeric and Legate Sparse), letting Diffuse find optimization opportunities across function and library boundaries. Diffuse accelerates unmodified applications developed by composing task-based libraries by 1.86x on average (geo-mean), and by between 0.93x–10.7x on up to 128 GPUs. Diffuse also finds optimization opportunities missed by the original application developers, enabling highlevel Python programs to match or exceed the performance of an explicitly parallel MPI library.

CCS Concepts: • Computing methodologies→Distributed programming languages.

Keywords: Distributed Programming; Composable Software

## ACM Reference Format:

Rohan Yadav, Shiv Sundram, Wonchan Lee, Michael Garland, Michael Bauer, Alex Aiken, and Fredrik Kjolstad. 2025. Composing Distributed Computations Through Task and Kernel Fusion. In Proceedings of the 30th ACM International Conference on Architectural

![](_page_0_Picture_15.jpeg)

[This work is licensed under a Creative Commons](https://creativecommons.org/licenses/by/4.0/) [Attribution International 4.0 License.](https://creativecommons.org/licenses/by/4.0/)

ASPLOS '25, March 30-April 3, 2025, Rotterdam, Netherlands © 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0698-1/25/03 <https://doi.org/10.1145/3669940.3707216>

Support for Programming Languages and Operating Systems, Volume 1 (ASPLOS '25), March 30-April 3, 2025, Rotterdam, Netherlands. ACM, New York, NY, USA, [16](#page-0-0) pages. [https://doi.org/10.1145/3669940.](https://doi.org/10.1145/3669940.3707216) [3707216](https://doi.org/10.1145/3669940.3707216)

