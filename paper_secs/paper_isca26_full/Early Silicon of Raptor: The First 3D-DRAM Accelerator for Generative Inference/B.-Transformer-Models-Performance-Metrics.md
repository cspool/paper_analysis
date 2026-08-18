# *B. Transformer Models: Performance Metrics*

Time-to-First-Token (TTFT) measures latency to generate a token in the prefill phase and reflects initial responsiveness. Time-per-Output-Token (TPOT) captures the cost of generating each token during decode [92]. Throughput is measured at two levels: The tokens-per-second-per-user (tok/s/usr ≈ 1/TPOT), while per-card throughput is tokens-per-second-percard (tok/s/card ≈ U/TPOT) for U concurrent users [89].

