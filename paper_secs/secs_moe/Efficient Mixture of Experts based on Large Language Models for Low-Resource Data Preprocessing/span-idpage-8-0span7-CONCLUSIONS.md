# <span id="page-8-0"></span>7 CONCLUSIONS

We proposed an e#cient Mixture of Experts on Large Language Models for Data Preprocessing (MELD) that is a universal solver for the low-resource DP tasks. To adapt to low-resource environment, we develop several expert-tuning and MoE-tuning techniques, including the RAG system, meta-path search strategy, expert re!nement and router network training. We also theoretically prove that MoE in MELD is superior than a single expert and the proposed router network is able to assign data to the right experts. Finally we conduct thorough experiments to show MELD outperforms state-of-the-art methods in aspects of e#ciency and e"ectiveness, especially in the low-resource environment.

In future work, we will explore the possibility to adapt MELD in multi-source setting with limited human annotation, and integrate such additional information into complex structures, e.g. graph. Also, the RAG in MELD could be replaced to !t for more complex scenarios, e.g. searching and retrieving relevant information over high-dimensional data spaces with vector database.

