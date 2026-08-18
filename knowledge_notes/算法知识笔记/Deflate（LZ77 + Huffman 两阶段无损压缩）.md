## Deflate（LZ77 + Huffman 两阶段无损压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Deflate 是广泛使用的无损压缩格式（zlib/gzip 的基础），顺序执行两个阶段：第一阶段用 LZ77 对数据做序列压缩（用 ⟨offset|length⟩ 对替换重复序列），得到字面量（literal）与序列标识符（sequence identifier）的混合流；第二阶段用 Huffman 符号压缩对该混合流编码——频繁符号用更短码、罕见符号用更长码。两阶段配合使 Deflate 在多样化数据上获得高压缩率（本论文实测软件 zlib 在 88 benchmark 内存 dump 上几何平均 3.84×）。Huffman 阶段使用符号字典映射字面量到码字，字典存在压缩输出中（与 LZ 不同）。
- 从算法pipeline角度拆解术语：压缩 pipeline 为 ① LZ77 匹配：在滑动窗口中找最长匹配，输出 ⟨offset|length⟩ 或字面量；② Huffman 编码：统计字面量与序列标识符的出现频率构建 Huffman 树，把各符号映射为变长码；③ 输出"码字流 + Huffman 字典"。解压 pipeline 必须从页/流首开始：先 Huffman 解码出字面量与 ⟨offset|length⟩，再按序从滑动窗口复制重建数据——服务中间任意位置的一个 64B 块需要解压其之前的所有数据（本论文的核心痛点）。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 硬件实现（本论文的 baseline）：TMCC [MICRO'22] 的 ASIC Deflate 面向内存压缩，用截断 Huffman（truncated Huffman）保证快速解压，16 符号字典、固定 hash 表、受限 lookahead，硬件设计约束使压缩率比软件低 ~10–12%。本论文对比：TMCC ASIC Deflate 半页解压延迟 140ns（2.5GHz、7nm 综合），RST 每 64B 块 18ns；压缩率 3.3×（Deflate）vs 3.4×（RST）。其他硬件：IBM Power9/z15 Deflate 加速器（~1µs）、OCP Project Zipline（open-source RTL，~2µs）、CDPU（~1µs）。软件：zlib/znzlib。
涉及论文标题：
- Random-Access Hardware Sequence Compression
