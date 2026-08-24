# <span id="page-21-0"></span>K Analysis of Reasoning Behaviros

We apply the cognitive behavior framework proposed by Gandhi et al. [\[5\]](#page-10-7) to conduct a detailed analysis of how reasoning behaviors change during our long-to-short RL. We use gpt-4.1-mini to perform a more fine-grained analysis of cognitive behaviors throughout the training process. Following Zeng et al. [\[28\]](#page-12-0), we use the prompt shown in Figure [12](#page-22-0) to prompt gpt-4.1-mini to identify and analyze reasoning behaviors. We analyze these behaviors on AIME2024 by sampling one question 16 times, resulting in 480 responses for analysis. Since we start from a LRM, reasoning behaviors such as backtracking naturally appear in every response, especially for challenging benchmarks. We specifically track four key behaviors: *Backtracking*, *Verification*, *Enumeration*, and *Subgoal Setting*. For each behavior, we calculate its frequency ratio relative to all behaviors and report how these ratios change throughout the training process. The complete list of all reasoning behaviors analyzed is provided in Table [7.](#page-21-2)

Table 7: Complete list of reasoning behaviors

