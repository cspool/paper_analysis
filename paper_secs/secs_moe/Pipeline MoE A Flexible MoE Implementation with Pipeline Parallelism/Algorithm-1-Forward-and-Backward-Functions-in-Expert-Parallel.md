# Algorithm 1: Forward and Backward Functions in Expert Parallel

```
def forward(hidden_states):
# All-reduce gradients in the backward pass
hidden_states = copy_to_tensor_parallel_region(hidden_states)
logits = wg(hidden_states)
scores = softmax(logits)
indices, weights, l_aux = gate_function(scores)
sliced_inputs = []
for i in range(num_local_experts):
    sliced_inputs.append(hidden_states.index_select(indices[i]))
expert_outputs = []
for i, expert in enumerate(experts):
    expert_outputs.append(expert(sliced_inputs[i]))
output_hidden_states = torch.zeros_like(hidden_states)
for i in range(num_local_expert):
    output_hidden_states[indices[i], ...] = expert_outputs[i]
# all-reduce output_hidden_states in the forward pass
output_hidden_states = reduce_from_tensor_parallel_region(
                                     output_hidden_states)
return output_hidden_states, l_aux
```

an all-reduce communication on gradients for each update (each global batch). Luckily, the communication overhead of an MoE layer for data and gradient gathering is exactly the same as tensor parallel, indicating that no extra communication overhead on data is introduced compared to tensor parallel. The only overhead is the synchronization of the linear mapping parameters in the gating module with a shape of h×E. Compared to the communication overhead on data *i. e.*, 2×b×s×h for each global batch, it is negligible since 2 × b × s >> E (4∼5 orders larger). In a word, almost no extra communication overhead compared to tensor parallel is required.

