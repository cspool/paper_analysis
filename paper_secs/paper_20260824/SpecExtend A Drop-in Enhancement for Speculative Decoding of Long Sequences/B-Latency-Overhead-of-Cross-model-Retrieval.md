# B Latency Overhead of Cross-model Retrieval

<span id="page-10-1"></span>

|              | Target  | Target Forward | Draft   | Retrieval Cache |
|--------------|---------|----------------|---------|-----------------|
|              | Forward | w/ Retrieval   | Forward | Update          |
| Latency (ms) | 53.76   | 54.11          | 0.84    | 0.34            |

Table 7: Latency overhead of a single retrieval cache update step on 16K token inputs.

## <span id="page-10-2"></span>C Experiment Details

The EAGLE models[1](#page-10-4) for vicuna-7b-v1.5-16k and longchat-7b-16k are trained on the ShareGPT dataset using default training settings with 4 A100 40GB GPUs. For each input length from 1K to 16K tokens, we sample 20 inputs, run each input twice, and report metrics averaged over all runs. We apply OPT-Tree's dynamic tree expansion strategy with the default settings of 50 total nodes, maximum depth 10, and threshold 0.7. We use the optimal working KV cache size and retrieval parameters described in Section [8.](#page-11-0)

