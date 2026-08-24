# C Problem Collection Framework

#### <span id="page-15-0"></span>C.1 Generating Performance Tests

Our benchmark construction pipeline's (described in Section [2.2\)](#page-2-0) effectiveness stems from two aspects: First, execution precisely identifies commits with consistent performance improvements across test cases. Second, as shown in Table [3,](#page-15-4) rich context from affected files and PRs yields gains in commits retained (pass functional equivalence checks and show performance improvement) for the benchmark. While a more sophisticated approach could be used (e.g., using SWE-agents [Yang et al.](#page-12-17) [\[2025\]](#page-12-17)) we

<span id="page-15-4"></span>

| Setting             | % Retained |
|---------------------|------------|
| Testgen             | 32.5       |
| + w/ Commit context | 43.3       |

Table 3: Rich commit context increases performance test quality and yield of retained commits after execution.

use a pipeline that uses sampling to scale tests for a large number of commits cost-effectively.

