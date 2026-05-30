## Answer-Think-Answer Template / Thinking Once, Answering Twice

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Answer-Think-Answer Template（"先答-再思-后审"，也称 Thinking Once, Answering Twice）是 VideoAuto-R1 提出的输出格式范式。模型在一次生成中按序输出三部分：(1) 第一轮 boxed 答案 $\boxed{a_1}$（简洁直接答案或 fallback 字符串）；(2) `<think>` 标签包裹的自由形式推理链 $r$；(3) 第二轮回 boxed 答案 $\boxed{a_2}$（审查后答案，可与 $a_1$ 相同或修正）。格式约束为恰好两个 `\boxed{}` 块和一个 `<think>...</think>` 块，无前后额外文本。核心理念：将"何时思考"（inference-time 早停决策）与"如何思考"（training-time RL 学到的推理行为）解耦。训练时模型学会同时输出两种答案并让两者都正确；推理时若初始答案足够自信则早停，否则继续推理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
System Prompt（Table 2）精确定义了输出格式：

```
SYSTEM PROMPT:
You are a helpful assistant.

FIRST: Output your initial answer inside the first \boxed{...} without any
analysis or explanations. If you cannot determine the answer without reasoning,
output \boxed{Let's analyze the problem step by step.} instead.

THEN: Think through the reasoning as an internal monologue enclosed within
<think>...</think>...

AT LAST: Output the final answer again inside \boxed{...}. If you believe
the previous answer was correct, repeat it; otherwise, correct it.

Output format: \boxed{...}<think>...</think>\boxed{...}
```

具体输出示例（来自 VideoMMMU 数学推理题）：
```
\boxed{D}  (confidence 0.92, continue CoT reasoning)
<think>
To find P(x < 3.5 | x < 4), use conditional probability:
P(A|B) = P(A∩B)/P(B). Since A ⊂ B, P(A∩B) = P(A).
f(x) = 1/(4.5-1.5) = 1/3
F(x) = (x-1.5)/3
P(x < 3.5) = 2/3, P(x < 4) = 2.5/3
P(x < 3.5 | x < 4) = (2/3)/(2.5/3) = 2/2.5 = 0.8
Therefore: C. 0.8
</think>
\boxed{C}
```
注意：初始答案是 D（错误），推理后修正为 C（正确）。

输出格式通过 strict regex check 的 format reward 强制：$R_{fmt} \in \{0,1\}$（二进制，完全符合模板得 1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 格式通过精心设计的 system prompt（Table 2）强制，无需 cold-start SFT 教格式——VideoAuto-R1 直接 RL 训练即可获得 ~100% format compliance；(2) Fallback 机制：当问题无法立刻回答时，$a_1$ 输出 "Let's analyze the problem step by step"，避免低置信度猜测，confidence score 被设为 $-\infty$ 强制继续 CoT；(3) 训练时 $w_2 > w_1$（如 0.9:1.1）确保推理后答案优先于初始答案获得更高 rewards。此模板的优势：解耦 training objective 和 inference policy，用户可按需选择始终使用 $a_2$（高精度）或早停 $a_1$（高效率），提供灵活的精度-效率 trade-off。

涉及论文标题：
- VideoAuto-R1__Video_Auto_Reasoning_via_Thinking_Once__Answering_Twice
