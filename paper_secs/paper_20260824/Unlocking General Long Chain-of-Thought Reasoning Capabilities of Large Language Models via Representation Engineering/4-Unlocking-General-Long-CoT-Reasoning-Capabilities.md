# 4 Unlocking General Long CoT Reasoning Capabilities

As discussed in Section 3.2, long CoT reasoning appears to be a general capability potentially encoded in LLMs. Therefore, it is feasible to unlock this capability through representation engineering with long-CoT data. However, not all domains are easy to obtain high-quality long CoT data. To solve this, our idea is to decouple long CoT reasoning into general reasoning patterns and domain-specific information. Since both reasoning patterns and domain-specific information are important for general long CoT reasoning, we design tailored methods to extract each kind of representation and inject them to control model behaviors. The overall framework is illustrated in Figure 3.

