![](_page_0_Picture_0.jpeg)

![](_page_0_Picture_1.jpeg)

![](_page_0_Picture_2.jpeg)

# **Marconi: Prefix Caching for the Era of Hybrid LLMs**

**Rui Pan**, Zhuang Wang, Zhen Jia, Can Karakus, Luca Zancato, Tri Dao, Yida Wang, Ravi Netravali

![](_page_0_Picture_5.jpeg)

![](_page_0_Picture_6.jpeg)

![](_page_0_Picture_7.jpeg)

![](_page_0_Picture_8.jpeg)

![](_page_0_Picture_9.jpeg)

![](_page_0_Picture_10.jpeg)

![](_page_0_Picture_11.jpeg)

![](_page_0_Picture_12.jpeg)

MLSys 2025, Santa Clara, CA

**Outstanding Paper Honorable Mention!**

![](_page_1_Figure_1.jpeg)

![](_page_2_Picture_1.jpeg)

• Quadratic compute complexity

![](_page_3_Picture_2.jpeg)

- Quadratic compute complexity
- Huge KV cache sizes (linear to sequence length)

|                             | Attention |
|-----------------------------|-----------|
| Computational<br>Complexity | O(L2)     |
| Inference-Time<br>Memory    | O(L)      |

![](_page_4_Picture_4.jpeg)

- Compress prior context into a state
- Update states recurrently in-place

|                             | Attention          |
|-----------------------------|--------------------|
| Computational<br>Complexity | O(L <sup>2</sup> ) |
| Inference-Time<br>Memory    | O(L)               |

![](_page_5_Figure_4.jpeg)

- Compress prior context into a state
- Update states recurrently in-place

|                          | Attention          | SSM  |
|--------------------------|--------------------|------|
| Computational Complexity | O(L <sup>2</sup> ) | O(L) |
| Inference-Time<br>Memory | O(L)               | O(1) |

![](_page_6_Figure_4.jpeg)

|                             | Attention | SSM  |
|-----------------------------|-----------|------|
| Computational<br>Complexity | O(L2)     | O(L) |
| Inference-Time<br>Memory    | O(L)      | O(1) |

- Memory consumption:
  - Fixed-sized regardless of num tokens

|                             | Attention | SSM  |
|-----------------------------|-----------|------|
| Computational<br>Complexity | O(L2)     | O(L) |
| Inference-Time<br>Memory    | O(L)      | O(1) |

![](_page_8_Figure_4.jpeg)

- Memory consumption:
  - Fixed-sized regardless of num tokens
  - Generally smaller than **whole sequences**' KVs

|                             | Attention | SSM  |
|-----------------------------|-----------|------|
| Computational<br>Complexity | O(L2)     | O(L) |
| Inference-Time<br>Memory    | O(L)      | O(1) |

![](_page_9_Figure_5.jpeg)

- Memory consumption:
  - Fixed-sized regardless of num tokens
  - Generally smaller than **whole sequences**' KVs
  - Orders of magnitude larger than a **single token**'s KVs

|                             | Attention | SSM  |
|-----------------------------|-----------|------|
| Computational<br>Complexity | O(L2)     | O(L) |
| Inference-Time<br>Memory    | O(L)      | O(1) |

![](_page_10_Figure_6.jpeg)

- A few Attention layers + many SSM layers
- Balances efficiency and language modeling capability Attention SSM

![](_page_11_Figure_3.jpeg)

- A few Attention layers + many SSM layers
- Balances efficiency and language modeling capability Attention SSM

![](_page_12_Picture_3.jpeg)

![](_page_12_Figure_4.jpeg)

![](_page_13_Figure_1.jpeg)

![](_page_14_Figure_1.jpeg)

#### **Execution Runtimes**

"Models == Transformers"

![](_page_14_Figure_4.jpeg)

![](_page_14_Figure_5.jpeg)

![](_page_14_Figure_6.jpeg)

![](_page_15_Figure_1.jpeg)

#### Background: prefix caching

- Reuses model states (KVs, SSM states) of common prefixes across requests
- Reduces Time To First Token (TTFT)

![](_page_16_Figure_3.jpeg)

• Prefix caching is challenging for SSMs: states can't be rolled back to represent a prefix

• Prefix caching is challenging for SSMs: states can't be rolled back to represent a prefix

KV Cache

• Prefix caching is challenging for SSMs: states can't be rolled back to represent a prefix

KV Cache

NYC is a busy city

• Prefix caching is challenging for SSMs: states can't be rolled back to represent a prefix

KV Cache

NYC is a

 Prefix caching is challenging for SSMs: states can't be rolled back to represent a prefix
SSM States

![](_page_21_Picture_2.jpeg)

 Prefix caching is challenging for SSMs: states can't be rolled back to represent a prefix

![](_page_22_Figure_2.jpeg)

 Prefix caching is challenging for SSMs: states can't be rolled back to represent a prefix
SSM States

![](_page_23_Figure_2.jpeg)

 Prefix caching is challenging for SSMs: states can't be rolled back to represent a prefix
SSM States

![](_page_24_Figure_2.jpeg)

SSM's modeling win complicates their systems win!

• Naive solution: checkpoint an SSM state every x tokens

![](_page_25_Figure_2.jpeg)

- Naive solution: checkpoint an SSM state every x tokens
- Catch 1: cache entries are sparsely-hit

![](_page_26_Figure_3.jpeg)

- Naive solution: checkpoint an SSM state every x tokens
- Catch 1: cache entries are sparsely-hit
- Catch 2: cache entries are huge

![](_page_27_Picture_4.jpeg)

- Naive solution: checkpoint an SSM state every x tokens
- Catch 1: cache entries are sparsely-hit
- Catch 2: cache entries are huge
- Frequent cache thrashing & low hit rate

![](_page_28_Figure_5.jpeg)

# **Marconi: prefix caching for Hybrid LLMs**

- Supports models with arbitrary layer compositions (Hybrid LLMs, pure Transformers, pure SSMs)
- Shouldn't focus solely on recency
  - Needs to be more judicious in admission and eviction!
- Leverages unique characteristics of Hybrid LLMs

Admission Eviction 10

#### Aside from recency:

# **Admission Eviction**

#### Aside from recency:

# **Admission Eviction**

Forecasts prefixes' reuse likelihoods

#### **Judicious admission**

- Existing systems: admit all states of most recent request
- Marconi: admit states with high reuse likelihood only
- Key insight
  - Future reuse patterns cannot be predicted…
  - …but can be sufficiently estimated through a taxonomy of potential prefix reusing scenarios!

### **Taxonomy of prefix reusing patterns**

• Composition of all reused prefixes:

# **Taxonomy of prefix reusing patterns**

- Composition of all reused prefixes:
  - 1. **Purely input**: part of the input sequence from a prior request
    - E.g., system prompts, few-shot examples

![](_page_34_Figure_4.jpeg)

(a) System prompt and few-shot prompting

# **Taxonomy of prefix reusing patterns**

- Composition of all reused prefixes:
  - 1. **Purely input**: part of the input sequence from a prior request
    - E.g., system prompts, few-shot examples
  - 2. **Input and output**: input+output sequence of a prior request
    - E.g., conversation history for chatbots, past environment interactions for agents

![](_page_35_Figure_6.jpeg)

#### **Different mechanisms for different cases**

![](_page_36_Figure_1.jpeg)

#### **Different mechanisms for different cases**

#### **• Purely input**

- Prefix shared by many requests
- Can be observed by bookkeeping and comparing previous requests

![](_page_37_Figure_4.jpeg)

(b) Multi-turn conversation (e.g., ChatGPT)

#### **Different mechanisms for different cases**

#### **• Purely input**

- Prefix shared by many requests
- Can be observed by bookkeeping and comparing previous requests

#### **• Input and output**

• Conversations usually append to the last decoded token

![](_page_38_Figure_6.jpeg)

- Use a radix tree to represent past requests
- Nodes naturally represent high reuse likelihood:

- Use a radix tree to represent past requests
- Nodes naturally represent high reuse likelihood:

![](_page_40_Figure_3.jpeg)

- Use a radix tree to represent past requests
- Nodes naturally represent high reuse likelihood:
  - Intermediates: purely-input prefixes

![](_page_41_Figure_4.jpeg)

- Use a radix tree to represent past requests
- Nodes naturally represent high reuse likelihood:
  - Intermediates: purely-input prefixes
  - Leaves: input-and-output prefixes

![](_page_42_Figure_5.jpeg)

#### Aside from recency:

# **Admission Eviction**

Forecasts prefixes' reuse likelihoods Considers compute savings hits deliver

### **Different memory-compute savings tradeoffs**

• Unlike KVs, SSM states have fixed size regardless of sequence length or compute savings

![](_page_44_Figure_2.jpeg)

# **Different memory-compute savings tradeoffs**

- Unlike KVs, SSM states have fixed size regardless of sequence length or compute savings
- **FLOP efficiency**: compute savings per unit of memory of reusing a state

![](_page_45_Figure_3.jpeg)

Total FLOPs across layers (Attn, SSM, MLP)

FLOP efficiency = Memory consumption of all states (KVs, SSM States)

# **Different memory-compute savings tradeoffs**

• Models with more SSM layers have more FLOP-efficient states

![](_page_46_Figure_2.jpeg)

FLOP efficiency = Total FLOPs across layers (Attn, SSM, MLP) Memory consumption of all states (KVs, SSM States)

#### **FLOP-aware eviction policy**

• Existing systems: recency-focused (i.e., evict using LRU)

Utility = recency

#### **FLOP-aware eviction policy**

- Existing systems: recency-focused (i.e., evict using LRU)
- Marconi: also considers the potential compute savings

Utility = recency

#### **FLOP-aware eviction policy**

- Existing systems: recency-focused (i.e., evict using LRU)
- Marconi: also considers the potential compute savings
- Utility score: balances recency and FLOP efficiency

Utility = recency + *α* ⋅ flop\_efficiency

#### **Evaluation**

- NVIDIA Mamba2-Hybrid-7B with {4, 24, 28} {Attention, SSM, MLP} layers
- Workloads: conversational (LMSys, ShareGPT) and agentic (SWEBench)
- Metrics: token hit rate (%), Time To First Token (ms)
- Large sweep of experiments with varying cache size and request arrival patterns

# **Marconi vs. fine-grained checkpointing**

- Judicious admission improves the cache utility significantly
- Average improvement in token hit rate: 4.5X, 7.3X, and 34.4X

![](_page_51_Figure_3.jpeg)

![](_page_52_Figure_2.jpeg)

![](_page_53_Figure_2.jpeg)

![](_page_54_Figure_2.jpeg)

![](_page_55_Figure_2.jpeg)

![](_page_56_Figure_2.jpeg)

![](_page_57_Figure_2.jpeg)

# **Marconi**

![](_page_58_Picture_1.jpeg)

![](_page_58_Picture_2.jpeg)

![](_page_58_Picture_3.jpeg)

![](_page_58_Picture_4.jpeg)

- First prefix caching system for models with arbitrary layer compositions
- Evaluates cache entries not only on recency, but also:
  - Admission: prefixes' reuse likelihoods
  - Eviction: compute savings that hits deliver
- Source code available!<https://github.com/ruipeterpan/marconi>

![](_page_58_Picture_10.jpeg)