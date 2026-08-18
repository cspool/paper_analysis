# *C. Description*

The code repository is based on the original ChampSim repository. The main branch of the repository is the one that has all the code and instructions used in this paper. A list of configurations is provided in ./sim\_configs. The code for existing prefetchers surveyed and the main implementation of R-Max is available in ./prefetcher.

R-Max relies on recording the memory traces seen at one or more cache levels in the previous iteration of the simulation and then replays it in the next iteration. For these recordings to remain coherent across simulations, page translations must remain static. In our design, prefetches are issued within the physical address space, but simulated workloads use fixed virtual addresses that require translations into non-fixed physical addresses. Depending on the order of arrival, the same addresses may have different translations as physical frames are allocated on a first-come, first-serve basis. By recording the page translations from the first run, we control this behavior and use a fixed set of page tables and translations for each workload across all simulations. To ensure accurate performance measurements, page fault penalties are still enforced by tracking whether a page has been accessed for the first time or not.

- ./README.md contains detailed instructions on what lines of code to modify to either record virtual to physical page translations or use recorded ones. But users can just invoke ./sim\_compile/compile\_all.sh to compile all configurations.
- *1) Access the code:* Please clone the provided GitHub repository using the provided link.
- *2) Software dependencies:* All dependencies are handled by vcpkg. Please see steps on how to setup ChampSim in ./README.md. Please note that any old version of Git may cause issues when trying to install packages via vcpkg. Please check the requirements on how to use vcpkg.
- *3) Data sets:* We do not use other data sets other than the ChampSim traces mentioned in the paper and this appendix.

