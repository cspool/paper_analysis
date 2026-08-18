# *B. Artifact Checklist*

- Algorithm: the paper presents the algorithm used in R-Max, it is broken down into the following components:
- 1) Fig. 2 shows the general workflow of R-Max.
- 2) Alg. 1 shows how to process the recorded memory traces to generate prefetches and replacement decisions.
- 3) Alg. 2 shows how to generate the dead block counters so that R-Max can keep track of the live time of each cache block.
- 4) Alg. 3 shows how to handle memory accesses in case of cache hit or cache miss.
- 5) Alg. 4 shows how to handle re-ordered memory accesses.
- Simulator: we use a modified version of the ChampSim simulator for our experiments, available on GitHub at ht tps://github.com/wilsonwang881/53rd ISCA 2026 R -Max Artifact. The repository is publicly available and contains the R-Max code as well.
- Compilation: the compilation scripts are provided. The code is tested to be compilable with GCC/11.3.0 and GCC/11.4.0.
- Source Code DOI: 10.5281/zenodo.19688265
- Code license: Apache-2.0 license.
- Program Traces: we use the same traces from the IPV based LLC replacement policy paper [26] which is available at 10.5281/zenodo.15298021 for the CVP-1 [30] traces. We have only evaluated the public set and have not evaluated the secret set of the CVP-1 traces due to the limited number of pages allowed. The SPECCPU 2017 traces are available at https://dpc3.compas.cs.stonybrook. edu/champsim-traces/speccpu/. The traces for GAP and XSBench are located at 10.5281/zenodo.20043527.
- Program Traces DOI: 10.5281/zenodo.15298021 for the CVP-1 traces. 10.5281/zenodo.20043527 for the GAP and XSBench traces. We do not have a DOI for the SPECCPU 2017.
- Data license: the traces are not firstly used in this paper and have been widely used in the research community.

