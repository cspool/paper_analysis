# **Author Contributions**

Christopher Clark, Jieyu Zhang, Zixian Ma, JaeSung Park, Rohun Tripathi, Sangho Lee and Mohammadreza Salehi collectively contributed to dataset construction, model training, and conducted numerous exploratory experiments for this project.

**Christopher Clark** led the project and focused on video modeling and training strategies, including experiments with the SFT mixture, the pre-training approach, and video modeling. He also wrote much of the core training code and implemented the packing and message tree systems.

**Jieyu Zhang** co-led the data effort on video datasets. He collected and filtered raw videos for Molmo2 video caption, video QA, and video pointing datasets, and contributed to the curation of these datasets. He helped the integration of other training/evaluation datasets and ran evaluations for many baseline models. He also helped add subtitle understanding to the model and ablations of the video SFT/caption models.

**Zixian Ma** co-led the data effort on video datasets. She designed human data collection interfaces and implemented them with help from Yinuo Yang. She collected the Molmo2-Cap, Molmo2-AskModelAnything, and Molmo2-VideoPoint datasets via Prolific. She led the training ablations on video counting and pointing and helped integrate academic training datasets. She ran the human preference and NLP evaluations.

**Jae Sung Park** led the effort to add tracking capability to Molmo2 as points. Together with Zhongzheng Ren and Vincent Shao, he designed the Molmo2-Track human annotation collection, curated existing academic tracking datasets for training, and built the pipeline to extract accurate point tracks. He introduced auxiliary grounding and single-point tracking objectives and performed ablations on mixtures of video tracking tasks. He and Zhongzheng Ren designed tracking evaluations across diverse VLMs and segmentation models.

**Mohammadreza Salehi** led the long-context post-training and co-led sourcing videos for training. He also contributed to training dataset construction, training on a mixture of images and videos, and evaluation of Molmo and API models.

**Rohun Tripathi** primarily worked on efficient modeling strategies. He developed learned and training free solutions to token allocation for different frames, with and without the input query. He implemented the initial training pipeline and details such as 3D position encoding and time tokens. He helped with training/evaluation set integrations, with a focus on long video understanding.

**Sangho Lee** led improvements to image modeling and training strategies and extended them to the multi-image setting. He also supported and directly conducted extensive ablation studies to develop effective training strategies for video modeling. In addition, he implemented the Hugging Face model and processor code and vLLM integrations.

**Chris Dongjoo Kim** led the data effort for multi-image datasets. In collaboration with Weikai Huang and Sangho Lee, he curated the MultiImageQA dataset. He also held full responsibility for the multi-image pointing capability, including dataset curation algorithms and model training.

**Yue Yang** led data curation for text-rich multi-image datasets, synthetically generating diverse question-answer pairs grounded in images such as charts, tables, and documents.

**Zhongzheng Ren**, **Yinuo Yang**, **Vincent Shao**, **Weikai Huang**, and **Ziqi Gao** all made significant dataset contributions.

**Jitesh Jain**, **Jianrui Zhang**, and **George Stoica** contributed to research discussions throughout the project and did exploratory experiments based on Molmo2.

**Taira Anderson** managed the project.

**Winson Han** designed the figures in this report.

**Ali Farhadi** advised the project.

**Ranjay Krishna** was the PI for the project.

