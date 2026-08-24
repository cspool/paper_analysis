# 3 Design

In this section, we present the design of ToolCaching. As discussed in [Section 2,](#page-1-1) caching in LLM tool-calling poses two unique challenges, and to address these issues, ToolCaching is designed around two core components:

- Feature Collection. ToolCaching first performs a comprehensive analysis of tool-calling requests by extracting both semantic attributes and system-level indicators. These features, obtained via LLM-based semantic understanding and lightweight system monitoring, provide the foundation for informed caching decisions.
- Caching Management Algorithms. Based on the collected features, ToolCaching employs an adaptive caching algorithm called VAAC (Value-Aware Adaptive Caching). VAAC integrates two key mechanisms: (i) selective cache admission guided by feature-based grouping and caching value (v-CACA), and (ii) multi-factor eviction (v-LRU) that jointly considers recency, frequency, and execution cost. This design enables ToolCaching to dynamically adapt to heterogeneous and evolving workloads.

The workflow of ToolCaching is shown in [Fig. 2.](#page-2-0) ToolCaching intercepts API requests from the LLM. If the request is already in the cache, ToolCaching retrieves the cached result and returns it to the LLM. If the request is new, ToolCaching forwards it to the tool API, extracts semantic features from the request, and determines its cacheability. If deemed cacheable, system-level features are collected and the caching value is evaluated. Based on this value and other features, ToolCaching decides whether to admit the result into the cache or to evict an existing entry. Regardless of whether the result is cached, it is returned to the LLM for further processing. If the request is not cacheable, ToolCaching simply forwards the request to the tool API and returns the result to the LLM.

In the following sections, we will discuss the details of each category and explain how we can overcome the complexities mentioned in [Section 2.](#page-1-1)

