## UI Navigation Graph (UNG)

术语是什么？

UI Navigation Graph (UNG) 是以图结构建模GUI应用导航关系的学术术语。UNG的节点是Windows UI Automation (UIA) 暴露的控件，边表示"点击一个控件后可达的新控件"的导航关系。与静态的UI accessibility tree不同，UNG捕获的是应用功能导航拓扑——哪些菜单/按钮/选项卡的点击会导致哪些新控件/面板/对话框出现。UNG的特点是包含环路（如多路径到达同一功能、反复切换tab）和merge node（不同路径汇聚到同一控件），这使得原始UNG不是树结构而是带环有向图。UNG用于Computer-Use Agent的确定性导航规划——将LLM从"猜测导航路径"解放出来，转为图上的确定性路径求解。

从算法pipeline角度拆解术语：

DMI中UNG的构建pipeline：

```
输入: 目标应用进程 + UIA root element
输出: UNG (节点=UIA控件, 边=点击后可达关系)

1. 初始化: root_node ← UIA.GetRootElement(app_window)
2. DFS构建:
   for each unexplored_node in current_tree:
       state_before ← UIA.CaptureTree()       // 记录当前控件树快照
       UIA.InvokePattern(unexplored_node).Invoke()  // 点击候选控件
       WaitForUIStabilization()                // 等待新UI出现
       state_after ← UIA.CaptureTree()         // 记录点击后控件树快照
       new_controls ← state_after - state_before
       for each new_ctrl in new_controls:
           add_edge(unexplored_node, new_ctrl) // 新出现的控件构成导航边
       if new_top_window_detected:             // 新顶层窗口
           push_to_stack(new_window)           // 递归探索
3. Differential capture处理:
   - 使用process_id + window listener检测新顶层窗口/modal window
   - 跟踪已访问节点，避免无限循环
4. 控件ID分配:
   - XPath-like格式: primary_id|control_type|ancestor_path
   - primary_id优先级: UIA automation_id > 控件名称 > "[Unnamed]"
```

关键设计：differential capture确保只记录"由点击触发的新出现控件"而非所有已有控件，过滤掉UI re-render的噪声。

术语一般如何实现？如何使用？

UNG构建在DMI中实现为自动探索脚本：(1) 对每个Office应用（Word/Excel/PowerPoint），自动化DFS探索 < 3小时；(2) 人工配置约1.5人日（标记access blocklist、选择context-aware exploration的代表对象）；(3) UNG构建结果包含5K+控件节点和对应的导航边；(4) UNG的环路和merge node导致从LLM角度看存在歧义路径，因此需要进一步转为path-unambiguous forest。UNG是实现policy-mechanism separation的基础数据——它将"如何导航"的知识编码为可供确定性算法查询的数据结构，而非让LLM在prompt中推理导航路径。

涉及论文标题：
- From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

---

