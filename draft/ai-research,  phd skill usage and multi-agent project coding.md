1、research-skill的optimization中提供的skill解析。
所有paper的科研方法流（搜索literature、发散、实验、循环迭代）。

2、phd skill中literature search的使用。
单个paper的辅助skill（gap、深入、复现）。

3、实验环境的设计，基于知识库探索，基本台子。
3.1我需要什么实验。
3.2有什么线程实验环境或代码。

4、实验project的build，多agent设计。每个agent一个role，侧重固定问答，多个agent的输入输出设计。
	按照quick-start快速达到目标后迭代。


当前目录下生成learning-experiment-from-notes的skill
1、question agent的要求
	1.1 用户指定学习方向，决定notes中的搜索方向（比如'单个GPU、NPU、加速器作后端，单请求模型推理（MoE、wavelet-Diffision、DiT、多模态、Video等较新模型）的负载场景下，	如何进行多算子/微算子的并发，垂直梳理方法、实验框架/环境。侧重硬件体系结构、编译框架的方法、对应实现和实验环境'。）
	1.2 对输入问题进行分层，构造分层的问题空间，每层包含用户的一部分输入和构造的若干问题（参考paper-experiment-idea的skill中算法、Serving、编译、kernel、硬件架构、芯片设计的分层标准）
	1.3 记录分层的问题空间（每个层次的问题，比如方法有什么？对应的实现？实验环境？）
	
2、answer agent的回答要求
	2.1 对要回答的问题，逐一调用obsidian api来搜索并加入上下文（参考obsidian-keyword-explain的skill中的方式，比如语义分割拆解、搜索和读取上下文）
	2.2 不同层次的问题回答要具体
		方法：模型具体到伪代码或计算过程，Serving和编译具体到框架运行模拟的例子，kernel调度具体到伪代码实现、指令和pipeline编排，硬件具体到数据流设计和计算、控制模块设计和功能说明，芯片具体到设计和评估。
		实现和实验和方法对应。
	2.3 工具、notes引用来确保证据

3、horizon summary的要求
	对所在层次的问题和答案进行分类和总结，但需要保持回答的具体程度，分类总结是减少冗余的问题和答案，但对分类后的每个类别的问答，仍需要足够具体。

4、vertical summary的要求
	对所有层次分类后的结果进行垂向梳理输出（如模型负载-Serving-编译-kernel调度-后端的不同组合下，负载定义到后端执行的全过程中，每个过程中涉及到方法、实现和对应实验环境是什么？）


检查下调度逻辑是否和下面对齐，并进行修正：

1、scheduler的ts脚本负责调度
脚本启动的agent都使用claude -p（不需要指定model），参考‘/data3/paper_analysis/scripts/run_all_papers.py’。
phase 1：设置workdir并建立所有agent所需的输出path，传入用户输入并开启question agent，等待question agent完成，基于每层的问题空间<layer_id_问题空间>.md（每个问题的存储格式是<question_id, layer_id, content>）构建工作池（entry的list，entry的数据结构是<question_id, layer_id, is_done>），完成后进入phase 2。
phase 2：设置3个worker线程，每个worker启动1个answer agent回答1个问题，按格式传入问题和输出path（<question_id_layer_id_answer>.md的path）给agent，worker等待agent正常结束后，更新对应问题entry的is_done，重新开启answer agent来回答下一个未完成问题，直到所有问题得到解答（工作池的所有entry都done）进入phase 3。
phase 3： 设置2个worker线程，每个worker线程启动1个horizon_summary agent，负责指定 layer的问题和回答的分类总结，传入输出layer_id_horizon_summary.md的path和问题空间path和对应所有的答案<question_id_layer_id_answer>.md的path，每个worker线程等待agent结束后，执行下一个未完成的layer，直到所有层的水平分类完成。
phase 4：启动1个vertical_summary agent，传入所有层的layer_id_horizon_summary.md的path和输出summary.md的path，直到agent结束。


2、四类agent
2.1 question agent对每层构建问题空间后将每层的问题空间输出到指定path的<layer_id_问题空间>.md（每个问题的存储格式是<question_id, layer_id, content>），question agent结束。
2.2 answer agent负责一个问题，模仿obsidian-keyword-explain的skill来回答问题，并将问题答案和所用上下文的来源输出到<question_id_layer_id_answer>.md，完成后当前answer agent结束。
2.3 每层分类的horizon_summary agent，接收属于layer_id的问题空间path和对应所有的答案md的path，读取后进行每层答案和问题的分类，输出到layer_id_horizon_summary.md。
2.4 垂向梳理的vertical_summary agent，接收所有层的layer_id_horizon_summary.md的path，读取后进行联系每层的垂向梳理，输出到summary.md。


输入：'单个GPU、NPU、加速器作后端，单请求模型推理（MoE、wavelet-Diffision、DiT、多模态、Video等较新模型）。如何进行多算子/微算子的并发实验来测试性能，垂直梳理方法、实验框架/环境。侧重硬件体系结构、运行时环境和编译框架的方法、实现和对应实验环境，实验环境尽量具体'。


brainstorm-skill
	按照用户输入进行迭代？
	迭代前记录
	每个层次基于知识库按照语义扩展
	目标定义迭代，每个扩展是一个迭代



记住先测试
repo_mdsplit_py_skill

修改为batch处理，原本是指定md的src path和out path，现在我希望你批量执行多个md的分割（每个md有独立的src和out path）。需求如下:

1、我会给你一个root_repo_path（比如'/data3/paper_analysis/repos/repo_model_quant'），它包含3个子目录对应3个repo（experiment_repo、idea_repo和knowledge_repo），每个repo包含至多6个md，这是你需要转换的6个src path。

2、每个repo下src md的out path的映射规则是，'experiment_repo'下的md输出到'/data3/paper_analysis/experiment_notes/'+exp_class_relative_path，'idea_repo'下的md输出到'/data3/paper_analysis/idea_notes/'，'knowledge_repo'下的md输出到'/data3/paper_analysis/knowledge_notes/'+know_class_relative_path。

3、exp_class_relative_path由src md的名字'<exp_name>'决定：
<exp_name>是'实验_编译框架'时，exp_class_relative_path='编译实验笔记'；
<exp_name>是'实验_算法pipeline'时，exp_class_relative_path='算法实验笔记'；
<exp_name>是'实验_芯片设计'时，exp_class_relative_path='芯片实验笔记'；
<exp_name>是'实验_硬件架构'时，exp_class_relative_path='硬件实验笔记'；
<exp_name>是'实验_kernel调度'时，exp_class_relative_path='kernel实验笔记'；
<exp_name>是'实验_Serving调度'时，exp_class_relative_path='系统实验笔记'；

4、know_class_relative_path由src md的名字<know_name>决定：
<know_name>是'知识库_编译框架'时，know_class_relative_path='编译知识笔记'；
<know_name>是'知识库_算法pipeline'时，know_class_relative_path='算法知识笔记'；
<know_name>是'知识库_系统架构'时，know_class_relative_path='系统知识笔记'；
<know_name>是'知识库_芯片设计'时，know_class_relative_path='芯片知识笔记'；
<know_name>是'知识库_硬件架构'时，know_class_relative_path='硬件知识笔记'；
<know_name>是'知识库_kernel调度'时，know_class_relative_path='kernel知识笔记'；
