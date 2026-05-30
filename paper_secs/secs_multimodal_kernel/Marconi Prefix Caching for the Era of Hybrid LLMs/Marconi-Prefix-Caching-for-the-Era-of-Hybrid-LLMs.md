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

