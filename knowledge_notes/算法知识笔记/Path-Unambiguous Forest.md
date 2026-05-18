## Path-Unambiguous Forest

术语是什么？

Path-Unambiguous Forest 是将带环和merge node的UI Navigation Graph (UNG) 转换为无歧义路径的森林结构的过程。原始UNG中存在两种歧义：(1) 环路——同一功能可通过多条不同路径到达（如通过File菜单和通过Quick Access Toolbar均可到达Save），等价路径使导航算法面临选择；(2) merge node——多条路径汇聚到同一控件（如不同tab下均有指向同一dialog的button），在图遍历中产生路径交汇。Path-Unambiguous Forest通过保留main tree（每个功能的canonical路径）和共享shared subtrees（merge node引用的可复用子树），使得从任意root到任意target的导航路径唯一确定——LLM只需声明目标ID，DMI执行确定性路径求解。

从算法pipeline角度拆解术语：

Path-Unambiguous Forest的转换pipeline：

```
输入: UNG (带环 + merge node的有向图)
输出: Path-Unambiguous Forest (main tree + shared subtrees)

1. 环路检测与消解:
   for each cycle in UNG:
       选择canonical entry作为main tree路径
       其余等价路径在core topology中标注但不作主路径
       消解策略: 优先保留更短/更稳定的路径

2. Merge node处理:
   for each node N with indegree > 1:  // merge node
       在main tree中保留N的一个出现位置
       将N及其子树从main tree中移除（避免重复）
       将N标记为shared subtree (entry_ref = 原始父节点路径)
       在各使用N的父节点处添加引用指针

3. Forest生成:
   - Main tree: root → ... target nodes (唯一canonical路径)
   - Shared subtrees: merge node + 其children (多父节点共享)
   - 每个共享子树节点携带entry_ref_id: 标识从哪个main tree节点进入subtree
```

Forest结构的关键属性：从root到任意控件的导航路径是path-unambiguous的——给定目标控件ID和可选的entry_ref_id，存在唯一的导航控件序列。

术语一般如何实现？如何使用？

在DMI在线执行中：(1) LLM声明目标控件leaf ID（整型），若在shared subtree中则附带entry_ref_id；(2) DMI executor在forest中执行确定性路径求解——从root沿main tree到entry point，再沿shared subtree到目标，整条路径是唯一确定的；(3) forest消除了"LLM不知道选择哪条路径"的歧义——baseline在使用UNG信息时SR反而从44.4%降到42.0%，而DMI通过forest+确定性solver使SR提升到74.1%，部分收益来自消除路径歧义。

涉及论文标题：
- From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

---

