# *B. Suppression of routing overhead*

Table [IV](#page-9-0) lists the geometric-mean routing overhead across all 216 cases (3 topologies × 6 ISAs × 12 programs) for each compiler, with per-benchmark details shown in Fig. [11.](#page-10-0) CANOPUS achieves the lowest routing overhead for every ISA-topology combination. Specifically, CANOPUS reduces average routing overhead by 16.06% in Ccount and 26.44% in Cdepth compared to SABRE, by 34.70% and 21.25% compared to TOQM, and by 19.89% and 20.72% compared to BQSKIT.

Notably, CANOPUS uniquely leverages the synthesis capabilities of more expressive ISAs. With CANOPUS, transitioning from CX to more powerful ISAs yields substantial routing overhead reductions—e.g., from 1.88× to 1.39× (−26%) on 1D chain, and from 1.38× to 0.99× (−28%) on 2D square for Ccount when equipped with ZZPhase\_ ISA—while baseline

TABLE IV AVERAGE (GEOMETRIC-MEAN) ROUTING OVERHEAD.

<span id="page-9-0"></span>

| Routing overhead |          | In terms of Ccount |      |                         |      | In terms of Cdepth |      |                         |      |
|------------------|----------|--------------------|------|-------------------------|------|--------------------|------|-------------------------|------|
| Topo             | ISA Type |                    |      | sabre toqm bqskit canop |      |                    |      | sabre toqm bqskit canop |      |
|                  | CX       | 2.26               | 3.07 | 2.27                    | 1.88 | 2.57               | 2.38 | 2.18                    | 1.81 |
|                  | ZZPhase  | 1.97               | 2.75 | 1.92                    | 1.7  | 2.22               | 2.15 | 1.91                    | 1.63 |
|                  | SQiSW    | 2.06               | 2.63 | 1.85                    | 1.73 | 2.32               | 2.08 | 1.84                    | 1.68 |
| Chain            | ZZPhase_ | 1.61               | 2.18 | 1.69                    | 1.39 | 1.82               | 1.72 | 1.66                    | 1.35 |
|                  | SQiSW_   | 1.72               | 2.25 | 1.68                    | 1.45 | 1.95               | 1.76 | 1.66                    | 1.4  |
|                  | Het      | 1.65               | 2.23 | 1.58                    | 1.43 | 1.86               | 1.76 | 1.56                    | 1.36 |
|                  | CX       | 2.37               | 2.82 | 2.59                    | 1.93 | 3.05               | 2.68 | 2.66                    | 2.08 |
|                  | ZZPhase  | 2.12               | 2.65 | 2.25                    | 1.74 | 2.77               | 2.52 | 2.26                    | 1.91 |
|                  | SQiSW    | 2.14               | 2.48 | 2.17                    | 1.72 | 2.71               | 2.43 | 2.28                    | 1.96 |
| HHex             | ZZPhase_ | 1.7                | 2.08 | 1.88                    | 1.4  | 2.2                | 2.0  | 1.96                    | 1.56 |
|                  | SQiSW_   | 1.78               | 2.09 | 1.98                    | 1.46 | 2.27               | 2.02 | 2.1                     | 1.66 |
|                  | Het      | 1.74               | 2.13 | 1.86                    | 1.43 | 2.25               | 2.05 | 1.98                    | 1.58 |
|                  | CX       | 1.64               | 2.18 | 2.06                    | 1.38 | 1.94               | 1.87 | 2.47                    | 1.49 |
|                  | ZZPhase  | 1.35               | 1.87 | 1.61                    | 1.16 | 1.63               | 1.61 | 1.94                    | 1.24 |
|                  | SQiSW    | 1.63               | 2.05 | 1.74                    | 1.34 | 1.89               | 1.81 | 2.02                    | 1.42 |
| Square           | ZZPhase_ | 1.16               | 1.55 | 1.43                    | 0.99 | 1.39               | 1.36 | 1.65                    | 1.09 |
|                  | SQiSW_   | 1.31               | 1.69 | 1.56                    | 1.11 | 1.54               | 1.47 | 1.83                    | 1.2  |
|                  | Het      | 1.18               | 1.58 | 1.36                    | 1.0  | 1.41               | 1.38 | 1.56                    | 1.09 |

methods exhibit much less pronounced improvements. This confirms that CANOPUS does not merely benefit from ISA rebase but actively exploits ISA expressiveness during routing.

The advantage of CANOPUS also lies in its unified optimization of both gate count and circuit depth. In contrast, SABRE and BQSKIT are primarily gate-count-driven, while TOQM specializes in optimizing depth. This bias manifests in measurable weaknesses. TOQM incurs the worst count overhead across nearly all configurations. For instance, on 1D chain with CX, TOQM's reaches 3.07× routing in terms of Ccount, more than 63% above that of CANOPUS (1.88×). Conversely, BQSKIT suffers severe depth overhead on 2D square topology, where its Cdepth-related routing overhead consistently exceeds those of all other compilers (e.g., 2.47× for CX versus 1.49× for CANOPUS).

Additionally, CANOPUS maintains consistently low overhead across all benchmarks, whereas every baseline fails on specific circuits. For instance, TOQM and BQSKIT cannot effectively manage the routing overhead for some structurally challenging circuits like qec9 and qram; BQSKIT struggles with bv even under expressive ISAs.

