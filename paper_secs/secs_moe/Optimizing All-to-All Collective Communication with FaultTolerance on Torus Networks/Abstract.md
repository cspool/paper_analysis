# Abstract

Large-scale distributed processing is extensively employed for large model inference and training, such as Deep Learning Recommendation Models (DLRMs) and Mixture-of-Experts (MoE) models. However, the All-to-All collective, with its complex point-to-point communication patterns and blocking nature, has become a major performance bottleneck in distributed DLRM and MoE accelerations. Moreover, the prolonged distributed processing often encounters link failures, which severely impact system efficiency, reliability, and cost. Unlike switched-based topologies, which support any-toany connections like Clos networks, All-to-All communication on torus networks can interfere with one another by sharing routing paths, creating critical performance limitations.

To address these challenges, we propose single-dimensional algorithm and multi-dimensional scheduling for all-to-all optimizations with fault tolerance on torus. In fault-free scenarios, we propose HalfRing algorithm and DimRotation scheduling. HalfRing utilizes bidirectional links to construct shortest communication path on a ring, while DimRotation allocates communication sequences of each data chunk across multiple dimensions to achieve full bandwidth utilization. In faulty scenarios, we introduce FoldedRing algorithm and MATE scheduling. FoldedRing facilitates fault-tolerant communication on a ring, while MATE accelerates communication on the faulty ring by leveraging available links from other dimensions. Our results show that, compared to the ring algorithm with pipeline scheduling, HalfRing, DimRotation and their combination can achieve average performance speedups of 1.56×, 1.45×, and 2.28×, respectively. For All-to-All with a single link failure, MATE can achieve a 1.37× speedup compared to ring-based fault-free conditions. When compared with state-of-the-art routing methods in

<sup>∗</sup>Corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

MICRO '25, Seoul, Republic of Korea

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-1573-0/25/10

<https://doi.org/10.1145/3725843.3756057>

TPUv4 clusters, our approach achieves 1.57× and 1.61× speedups for fault-free and fault-tolerant scenarios, respectively.

