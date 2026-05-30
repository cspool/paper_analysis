# A.3 Description

A.3.1 How to access. The artifact is archived in Zenodo[1](#page-14-18) . It can also be accessed from GitHub, as the command shown below:

\$ git clone https :// github . com / redbird - arch / micro2025 - torus - ft - all2all - artifact . git

A.3.2 Hardware dependencies. For reference, we list our system configurations here:

For simulation experiments:

- OS: Ubuntu 22.04.5 LTS
- CPU: Intel(R) Xeon(R) Gold 6348H CPU @2.30GHz (24 cores); Other CPU would work.
- DRAM: 512 GB
- Disk: 2 TB

For real machine experiments:

- OS: Ubuntu 22.04.5 LTS
- NPU: 16×Ascend 910B4 NPUs
- NPU Memory: 16×32 GB
- Disk: 2 TB

A.3.3 Software dependencies. We ran our experiments on the Ubuntu 22.04 LTS operating system, but other versions of Ubuntu should also work. A Python runtime environment constitutes the fundamental requirement for operation for simulation experiments. For real-machine tests, the dependencies are CANN 8.2.RC1, torch\_npu == 2.1.0.post12. Complete dependency specifications are documented in the README.md file.

<span id="page-14-18"></span><sup>1</sup>https://doi.org/10.5281/zenodo.16735313

