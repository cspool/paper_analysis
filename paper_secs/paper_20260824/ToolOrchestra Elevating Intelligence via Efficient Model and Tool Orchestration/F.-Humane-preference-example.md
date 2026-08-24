# **F. Humane preference example**

**Tools**; = [ Web search, local search, Qwen/Qwen3-235B-A22B, meta-llama/Llama-3.3-70B-Instruct, o3-mini, o3 ]

Preference instruction:  = I am a company employee and there is some confidential information in my server. There are many GPUs in the server, so I can host open-sourced models or retrievers. It would be great if we can avoid API calling as much as possible.

Preference vector: = [0,1,1,1,0,0,0,0,0] Explanation: The first digit in the preference vector corresponds to the first tool in ; The second digit in the preference vector corresponds to the second tool in , etc. The last three digits in corresponds to accuracy, cost and latency, aligned with the definitions in [§3.2.](#page-3-2) Therefore, this preference vector means the user prefers to use local search, Qwen/Qwen3-235B-A22B, meta-llama/Llama-3.3-70B-Instruct.

