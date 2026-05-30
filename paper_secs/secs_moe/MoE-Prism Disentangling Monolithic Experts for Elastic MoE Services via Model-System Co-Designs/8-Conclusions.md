# 8 Conclusions

We introduced MoE-Prism, a complete model-system codesign that transforms static MoE models into truly elastic services. MoE-Prism operates in two phases: an Offline Refactoring Engine uses a partitioning optimization solver to deconstruct monolithic experts into fine-grained, functionally coherent sub-experts without costly retraining. This architectural elasticity is then exploited by the Online Scheduling Engine, an online component that implements utilitydriven policies to navigate the expanded configuration space and meet diverse system objectives. Ultimately, MoE-Prism bridges the gap between static model architectures and the dynamic demands of real-world serving systems and paves the way for the next generation of QoS-aware AI services.

