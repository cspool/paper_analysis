## imec.netzero

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
imec.netzero 是 imec SSTS（Sustainable Semiconductor Technologies and Systems）项目发布的公开"虚拟晶圆厂"web 应用（2023-11 公开版上线），用 LCA 方法自底向上建模 IC 制造的环境影响：从工艺流、recipe、机台工具数据构建高量产 fab 的排放清单，输出气候影响（kgCO2eq）与用水等指标；数据经 Air Liquide、Applied Materials、ASM、ASML、Lam、TEL、三星、台积电等合作伙伴持续对标。公开版覆盖 Scope 1+2 与 N28 及更先进逻辑节点（论文口径 N65–N2，EUV 自 N7 引入）及 DRAM/NAND。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 CAPA 中 imec.netzero 是逻辑 die 碳模型的参数源与验证基准：S1PW、EPW、CI_fab、S3PW 均取自其数据（Eqn. 4-5 的 CPW 分解）；验证时 CAPA 把逻辑 die 碳归一化到 imec.netzero 输出（Fig. 11），最大误差 3.23%，远优于 ACT（55% 低估）、3D-Carbon（118% 高估）、ECO-Chip（76–338%）——原因是 imec.netzero 以晶圆为单位建模（CPW 固定）而 ACT 按碳/面积线性缩放、3D-Carbon 用整片晶圆面积×碳/面积放大 CPW。其局限：公开版不支持 binning、封装、集成或 DRAM，这正是 CAPA 补位之处（Table I）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用：netzero.imec-int.com 交互式查询各节点/制程的 GHG 与水足迹；其 LCA 结论（3nm 逻辑中光刻+刻蚀占 Scope 1+2 约 45%、EUV 优于多重曝光）被 CAPA 引为制程层碳趋势背景。实现上作为上游数据工具被 ACT/ECO-Chip/3D-Carbon/CAPA 等架构碳模型普遍引用，是"架构层碳建模"与"制程层 LCA 数据"之间的接口。

涉及论文标题：
- CAPA: Manufacturing Carbon Estimation for Advanced-Packaged Architectures
