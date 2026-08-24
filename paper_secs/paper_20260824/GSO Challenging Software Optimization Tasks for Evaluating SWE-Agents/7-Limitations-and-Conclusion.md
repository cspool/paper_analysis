# 7 Limitations and Conclusion

Benchmark Size. Our benchmark contains 102 software optimization tasks, which may introduce variance in results due to its limited size. Nevertheless, each task represents a challenging real-world optimization problem, making successful completion a strong indicator of model capabilities for high-performance software development. We will consider expanding the benchmark based on community feedback, identifying additional representative tasks.

Hacky Optimizations. Reward hacking plagues software agent benchmarks [\[Gu et al.,](#page-10-15) [2025\]](#page-10-15) with agents circumventing test cases in unintended ways [\[Lange et al.,](#page-11-15) [2025\]](#page-11-15). As noted in Section [5,](#page-6-0) models already attempt to overfit tests and produce non-idiomatic code. Our precise task specifications and test suite currently detect such issues, but monitoring these behaviors remains critical for future work, and we recommend community efforts to develop mitigation approaches.

Evaluation Beyond Speedup. Our work focuses on improving the runtime performance of the code, but practical software development also requires other metrics such as memory usage, maintainability, and idiomaticity. For example, optimization often requires trade-offs between different metrics, which are not captured by our speedup metric. Unfortunately, automated evaluation of these properties is challenging, and we hope to tackle these challenges in future work.

Contamination. The current low performance suggests contamination is not a risk for existing LLMS despite our tasks being collected from GitHub repositories. Additionally, as discussed in Section [2.4,](#page-4-2) our continuous speedup metric helps detect contamination, as agent solutions that exceed human performance demonstrate generalization beyond mere memorization.

Conclusion. We present GSO, a benchmark for evaluating LLMS in aiding the development of high-performance software. Our quantitative results demonstrate that current LLMS fall short in this domain and our qualitative analysis identifies various failure modes. We hope GSO can serve as a valuable resource for future works in this direction in building more capable SWE-Agents, including improvements to both the model and the agent scaffold.

