# *F. Profiling Overhead.*

The online profiling phase is the only period during which PowerWeave introduces measurable latency overhead, as it sweeps kernels through multiple frequency points. However, profiling is spread across requests: kernels are profiled at different frequency points in different requests, so no single request experiences the full cost. The profiling window consists of two cycles of twelve frequency steps (from 1965 MHz to 915 MHz), spanning ≈150 requests, depending on the model. As a result, profiling does not cause any SLO violations across our experiments. PowerWeave can further control profiling aggressiveness: more conservative deployments can distribute profiling across more requests, reducing per-request impact at the cost of a longer profiling window.

