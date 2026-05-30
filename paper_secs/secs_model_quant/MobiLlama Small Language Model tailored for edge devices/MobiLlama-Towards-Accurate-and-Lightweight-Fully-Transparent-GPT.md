# MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT

Omkar Thawakar<sup>1</sup>\* , Ashmal Vayani<sup>1</sup>\* , Salman Khan<sup>1</sup>,<sup>2</sup> , Hisham Cholakal<sup>1</sup> , Rao M. Anwer<sup>1</sup>,<sup>3</sup> , Michael Felsberg<sup>5</sup> , Tim Baldwin<sup>1</sup>,<sup>4</sup> , Eric P. Xing<sup>1</sup> , Fahad Shahbaz Khan<sup>1</sup>,<sup>5</sup>

<sup>1</sup>Mohamed bin Zayed University of AI, <sup>2</sup>Australian National University, <sup>3</sup>Aalto University <sup>4</sup>The University of Melbourne, <sup>5</sup>Linköping University

## Abstract

'*Bigger the better*' has been the predominant trend in recent Large Language Models (LLMs) development. However, LLMs do not suit well for scenarios that require on-device processing, energy efficiency, low memory footprint, and response efficiency. These requisites are crucial for privacy, security, and sustainable deployment. This paper explores the '*less is more*' paradigm by addressing the challenge of designing accurate yet efficient Small Language Models (SLMs) for resource constrained devices. Our primary contribution is the introduction of an accurate and fully transparent open-source 0.5 billion (0.5B) parameter SLM, named *MobiLlama*, catering to the specific needs of resource-constrained computing with an emphasis on enhanced performance with reduced resource demands. *MobiLlama* is a SLM design that initiates from a larger model and applies a careful parameter sharing scheme to reduce both the pre-training and the deployment cost. Our work strives to not only bridge the gap in open-source SLMs but also ensures full transparency, where complete training data pipeline, training code, model weights, and over 300 checkpoints along with evaluation codes is available at : [https://github.com/](https://github.com/mbzuai-oryx/MobiLlama) [mbzuai-oryx/MobiLlama](https://github.com/mbzuai-oryx/MobiLlama).

