## Expert Combination Space in MoE（MoE 中的专家组合空间）

术语是什么？
Expert Combination Space 指 MoE 模型中 router 可选择的 expert 组合总数 C(E_routed, k_routed) (ICLR 2026)。与局部路由一致性正相关：组合空间越大，router 在相邻 token 间做局部微调的灵活性越高。Shared experts 缩减组合空间的双重机制：(1) 占用激活 quota（k_routed = k - shared）；(2) bypass effect——更多信息由 shared expert 处理，routed expert 重要性降低。

从算法pipeline角度拆解术语：
```
无 shared: C(64, 8) ≈ 4.4×10^9
2 shared (DeepSeekMoE): C(62, 6) ≈ 6.1×10^7 (缩小 ~72×)
TOY 实验验证: ActMore (C(64,16)=4.9e14) 提升 SRP; ActFewer (C(64,2)=2016) 降低 SRP
```

术语一般如何实现？如何使用？
架构设计指导：使用更多 total experts + 适中 k（不超过半数），避免 shared experts 过度占用配额。虽然此因素对 SRP 影响弱于 load balance 和 shared experts，但确实存在正相关。论文代码 https://github.com/ljcleo/moe-lrc。

涉及论文标题：
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

---
