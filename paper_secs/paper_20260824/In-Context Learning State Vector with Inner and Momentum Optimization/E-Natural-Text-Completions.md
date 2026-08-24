# E Natural Text Completions

In this study, we evaluate the effectiveness of the momentum optimized state vector on natural text completions. Given a natural text template, we instruct the model to greedily generate 5 tokens with or without intervention in the zero-shot setting. We use exact match accuracy as the metric. Table [3](#page-14-0) shows the result of natural text completions on Llama-2. The performance boosts observed with the momentum-optimized state vector on the separate tokens indicate that it can guide the model to generate answers correctly. We include more examples of natural text completions in the Appendix.

### F Case Study

In this section, we present a case study shown in Table [4,](#page-14-1) to demonstrate the efficacy of the momentumoptimized state vector in natural text completions. Consider the query: "What is the meaning of biography?", The vanilla Llama-2 model would directly answer this question. However, when influenced by an English-French state vector, Llama-2 changes its response, translating the question into French instead. Similarly, when presented with the sentence "When I think of upright, I think of". Influenced by an Antonym state vector, Llama-2 completes the sentence with an anonymous pattern. These instances exemplify the model learning the ICL function stored in the momentum optimized state vector, enabling it to generate context relevant to the specified task.

