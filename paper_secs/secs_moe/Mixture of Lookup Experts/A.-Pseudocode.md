# A. Pseudocode

### A.1. Training Phase

```
class MoleDecoderLayer(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.self_attn = Attention(config)
      self.shared_expert = MLP(config)
      self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
      self.routed_expert = nn.ModuleList([MLP(config) for _ in config.num_experts])
      self.input_layernorm = RMSNorm(config.hidden_size)
      self.post_attention_layernorm = RMSNorm(config.hidden_size)
      self.expert_layernorm = RMSNorm(config.hidden_size)
   def forward(self, hidden_states, embedding_states):
      '''Attention'''
      residual = hidden_states
      hidden_states = self.input_layernorm(hidden_states)
      hidden_states = self.self_attn(hidden_states)
      hidden_states = residual + hidden_states
      '''Shared Expert'''
      residual = hidden_states
      hidden_states = self.post_attention_layernorm(hidden_states)
      shared_output = self.shared_expert(hidden_states)
      '''Routed Expert'''
      router_value = nn.functional.softmax(self.router(hidden_states), dim=-1)
      embedding_states = self.expert_layernorm(embedding_states)
      routed_output = torch.stack([expert(embedding_states) for expert in
         self.routed_expert], dim=2)
      routed_output = (routed_output * router_value.unsqueeze(-1)).sum(dim=2)
      hidden_states = residual + shared_output + routed_output
      return hidden_states
```

### A.2. Inference Phase

```
class MoleDecoderLayer(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.self_attn = Attention(config)
      self.shared_expert = MLP(config)
      self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
      self.lut = LookupTable(config.vocab_size, config.num_experts * config.hidden_size)
      self.input_layernorm = RMSNorm(config.hidden_size)
      self.post_attention_layernorm = RMSNorm(config.hidden_size)
   def forward(self, hidden_states, input_ids):
      '''Lookup'''
      lookup_results = self.lut(input_ids).to(hidden_states.device, non_blocking=True)
      '''Attention'''
      residual = hidden_states
      hidden_states = self.input_layernorm(hidden_states)
      hidden_states = self.self_attn(hidden_states)
      hidden_states = residual + hidden_states
      '''Shared Expert'''
      residual = hidden_states
```

#### Mixture of Lookup Experts

```
hidden_states = self.post_attention_layernorm(hidden_states)
shared_output = self.shared_expert(hidden_states)
'''Routed Expert'''
router_value = nn.functional.softmax(self.router(hidden_states), dim=-1)
lookup_results = lookup_results.view(-1, config.num_experts, config.hidden_size)
routed_output = (lookup_results * router_value.unsqueeze(-1)).sum(dim=2)
hidden_states = residual + shared_output + routed_output
return hidden_states
```

