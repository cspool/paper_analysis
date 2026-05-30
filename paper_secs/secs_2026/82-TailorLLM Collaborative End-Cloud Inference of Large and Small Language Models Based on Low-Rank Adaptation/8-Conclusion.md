# 8 Conclusion

To address the rising costs of large language model inference services, this paper proposes TailorLLM, a task-level endcloud collaborative inference framework. Leveraging the long-tail and temporal patterns of user requests, TailorLLM enhances end-side SLMs with dynamically updated LoRA modules for frequent tasks while offloading complex queries to cloud LLMs. We introduce two key algorithms: RFLoRA for compact parameter tuning and AdapterMgr for adaptive LoRA management via imitation learning. Finally, we built an end-cloud prototype system and validated the effectiveness of our approach through evaluations on public datasets.

