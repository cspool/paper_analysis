## 指针加密 modifier（m1/m2 上下文绑定）

术语解释
LIPPEN 中把指针加密绑定到执行上下文的可配置参数 m = m1||m2：m1 XOR 进明文（放在不影响地址生成的未用位，|m1|max = 64-A+2），m2 XOR 进密钥（防跨域 key-collision），用 SET_M_SIZE 配置；等价于 PAC 的 modifier 概念，无需 tweakable cipher。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 密码学定义：seal(k, ptr, m) = Enc_{k⊕m2}(ptr ⊕ m1)；unseal = Dec_{k⊕m2}(cipher) ⊕ m1。三个组成部分：系统管理的秘密密钥 K（软件不可见）、指针值、上下文 modifier m。modifier 把每个加密指针绑定到其创建环境（栈帧、特权级、地址空间标识、控制流 epoch 等），防止指针被复制/泄露后在别的上下文复用（跨域指针重用攻击）。
- 设计动机：PRINCEv2 无 tweak 结构，直接把 modifier 混入明文/密钥（而非用更贵的 tweakable cipher 如 QARMA）可降低延迟与面积——这是"系统与密码协同设计"（co-design）的体现；SET_M_SIZE(m1,m2) 由系统管理器配置。
- 安全约束：① m1 只能放不影响地址生成的位（A-bit 地址空间外的高位 + 对齐指针的低位），否则 m1 与指针重叠可被构造 bit-flip 攻击（m1,a = m1,v⊕X、p_a = p_v⊕X 得合法密文）；② 域分离密钥的未受影响部分（128-|m2| bit）必须跨域唯一，否则攻击者可选 m2 使派生密钥撞上受害域密钥——域数上界 2^(128-|m2|)；③ 实践中 |m|=16 bit 足够（xalancbmk 最坏 ~32,097 指针变量/2,558 类型 ≈ 16 bit 唯一性；RSTI 14,073 上下文 ≈ 14 bit），故 LIPPEN 配 M1=16、M2=0。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件流程：PTR_SEAL(ptr, mod) 把 mod 拆成 m1（XOR 明文）与 m2（XOR 密钥）→ PRINCEv2 单周期加密 → 密文写回；PTR_UNSEAL 逆过程并校验 m1 位为 0。返回地址保护用 SP 作 modifier（对齐 PAC 的 RETAA）；数据指针保护可传零/共享非零/每访问加载的 modifier（微基准三种配置，实测 modifier 加载与 load-use 链独立可重叠、L1 命中开销可忽略）。
- 与 MTE/地址宽度扩展的兼容：地址空间 A 或 tag 字段扩大时 m1 预算缩小，可转用 m2 保持安全裕度（域分离空间 E = 128-|m|-|Tag|+(64-A)，A=48 时 124 bit、A=57 时 115 bit）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：SET_M_SIZE 指令配置长度；PRINCE/PRINCEv2 HDL 加入 m1/m2 XOR 逻辑（+mod 变体，1,522 LUT/42 MHz）；编译器按策略生成 modifier（如 PacTight 的强唯一 modifier 经 IR pass 移植，<50 行）。
- 使用：系统管理员/OS 设密钥与 m 配置；编译器/用户按保护策略选 modifier 来源（SP、类型 id、函数 id、随机 tag 等，Table V 列出 PARTS/PACStack/PTAuth/PACMem/AOS/PacTight/RSTI 的 modifier 设计）。

涉及论文标题：
- LIPPEN: A Lightweight In-Place Pointer Encryption Architecture for Pointer Integrity
