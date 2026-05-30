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

