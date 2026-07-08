# **Span Queries**

Nick Mitchell, Paul Castro, Nathan Ordonez, Thomas Parnell, Mudhakar Srivasta, Antoni Viros i Martin

**IBM Research | May 2026**

### **Pages (a.k.a. Blocks)**

**Contiguous sequence of tokens in a paged attention server**

![](_page_1_Figure_2.jpeg)

vLLM virtualizes **discontiguous physical** pages into **contiguous virtual** pages

So that they can be laid out in any physical order

![](_page_1_Figure_5.jpeg)

### **Spans**

#### **Contiguous sequence of pages in a paged attention server**

![](_page_2_Figure_2.jpeg)

**GPU memory** This extra level of indirection lets us reason about **spans as occurring in any order**.

![](_page_2_Picture_4.jpeg)

### **Commutativity Hypothesis**

**If order doesn't matter, we can optimize**

![](_page_3_Figure_2.jpeg)

AB=BA 㱺 **Commutativity**

# Q1: What does commutativity buy?

![](_page_4_Figure_1.jpeg)

Because we can **reuse pages** even if they were originally used in different context

![](_page_4_Picture_3.jpeg)

"Attention Locality"

Because we can apply **divide** and conquer to lost-in-the-middle problems.

# Q2: How do clients express a "span table"?

![](_page_5_Picture_1.jpeg)

#### **500 lines changed**

- 1. RoPE on read, not on write
- 2.Prefix scans must selectively disable block hash chaining for "spanned" regions

![](_page_5_Picture_5.jpeg)

Clients issue span queries, vLLM parses, plans, optimizes

### **The Gist**

#### **Generalize token sequences**

#### **Chat Completion**

![](_page_6_Picture_3.jpeg)

**API of Message Lists**

### **The Gist**

**Generalize token sequences**

**Chat Completion**

**Span Query**

![](_page_7_Picture_4.jpeg)

![](_page_7_Picture_5.jpeg)

**API of Message Lists API of Message Trees**

### **The Gist**

#### **Generalize token sequences**

**Chat Completion**

**S A U** *System message User message Output of a prior chat loop*

**Span Query**

![](_page_8_Picture_5.jpeg)

**API of Message Lists API of Message Trees**

#### **Chat use case**

![](_page_9_Picture_2.jpeg)

#### **RAG use case**

![](_page_10_Picture_2.jpeg)

#### **RAG use case with retrieval**

![](_page_11_Picture_2.jpeg)

#### **2-way Judge-Generator use case**

![](_page_12_Figure_2.jpeg)

#### **4-way Judge-Generator use case**

![](_page_13_Figure_2.jpeg)

4-way Judge-Generator query, optimized to avoid LITM

![](_page_14_Picture_2.jpeg)

### **Span Query Operation**

**Optimization and Execution**

![](_page_15_Figure_2.jpeg)

### **Accuracy**

#### **With span table implementation (MSMARCO)**

![](_page_16_Figure_2.jpeg)

### **Accuracy**

#### **With span table implementation (HotpotQA)**

![](_page_17_Figure_2.jpeg)

### **TTFT Speedup**

#### **RAG use case**

![](_page_18_Figure_2.jpeg)

### **TTFT Speedup**

**Judge/generator use case**

![](_page_19_Figure_2.jpeg)

### **Attention Locality**

**Needle in haystack experiments**

![](_page_20_Figure_2.jpeg)

### **Thanks!**

![](_page_21_Picture_1.jpeg)

<https://github.com/IBM/spnl>

![](_page_21_Picture_3.jpeg)

@starpit.bsky.social

Nick Mitchell, Paul Castro, Nathan Ordonez, Thomas Parnell, Mudhakar Srivasta, Antoni Viros i Martin

IBM Research | May 2026

### **Accuracy**

#### **With cropping implementation**

![](_page_22_Figure_2.jpeg)

## Backup Material on Tokenization

### Chat: Token Sequences

After <u>first</u> request

![](_page_24_Figure_2.jpeg)

Blocks in a paged attention inference server such as vLLM

e.g. 2 tokens per block

### Chat: Token Sequences

After <u>first</u> request

#### TODOs

Tally of features we need to support with span queries

![](_page_25_Figure_4.jpeg)

Blocks in a paged attention inference server such as vLLM

### **Chat: Token Sequences**

**After second request**

![](_page_26_Figure_2.jpeg)

![](_page_26_Figure_3.jpeg)

Hits because of prefix caching

vLLM doesn't cache partial blocks, who cares about 1/N misses

### RAG: Token Sequences

After <u>first</u> request

![](_page_27_Figure_2.jpeg)

![](_page_27_Figure_3.jpeg)

![](_page_27_Figure_4.jpeg)

![](_page_27_Figure_5.jpeg)

### RAG: Token Sequences

After <u>second</u> request

![](_page_28_Figure_2.jpeg)

![](_page_28_Figure_3.jpeg)

### RAG: Token Sequences

After selectively deleting positions

![](_page_29_Figure_2.jpeg)

![](_page_29_Figure_3.jpeg)

### Judge/Generator: Token Sequences

After two generators

![](_page_30_Figure_2.jpeg)

![](_page_30_Figure_3.jpeg)

Partially filled block, resulting in mixed block content, but maybe who cares because 1/N...

### Judge/Generator: Token Sequences

Two generators and one judge

![](_page_31_Figure_2.jpeg)

![](_page_31_Figure_3.jpeg)

### Judge/Generator: Token Sequences

![](_page_32_Figure_1.jpeg)

![](_page_32_Figure_2.jpeg)

### **Low-level Optimizations**

**Token-level transformations required to work with vLLM**

We may need to reintroduce prefixes for agentic workloads

We may need to pad or crop assistant output

**TODOs ◼ Prefix caching ◼ Partial blocks ◼ Position change**

We need to allow for the expression of commutativity, and be able to selectively delete positions in commutative intervals.

![](_page_34_Picture_0.jpeg)

### **The Use Cases**

**RAG**

![](_page_34_Picture_3.jpeg)

Use cases are evolving.

The **inference API needs to evolve** along with them.

![](_page_34_Picture_6.jpeg)