## LZ 序列压缩（LZ77/LZW/LZMA Lempel-Ziv 字典压缩家族）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LZ（Lempel-Ziv）是一族基于"序列（symbol sequence）替换"的无损压缩算法：把重复出现的变长符号序列（通常 3–258 个连续字节）用较短的"序列标识符"代替，从而获得高压缩率。核心机制（本论文背景部分）：先用更早出现的序列作为字典压缩更晚的数据。LZW：把首次遇到的唯一序列按出现顺序加入字典（如 "XY"→1*、"YZ"→2*），之后遇到字典中已有的序列就用其索引替换；为区分字典索引与字面量，LZW 输出用 9-bit 符号（8-bit 只能编码 256 个字面值，9-bit 可额外编码字典索引）。LZ77：用 ⟨offset|length⟩ 对作为序列标识符，offset 指向滑动窗口中该序列首次出现的位置、length 为其长度，从而让输入早期出现的重复长序列也能被压缩（克服 LZW 需先选短序列再选长序列的顺序限制）。LZMA 是 LZ77 的高压缩比变体（更大窗口+范围编码）。压缩率高的关键在"重复的变长序列"，内存数据中的重复指针、零初始化变量、模式化数据是典型冗余来源。
- 从算法pipeline角度拆解术语（本论文 Fig.1 的 LZW 例子）：输入 "XYXYZXYZYX..." 这类数据时，压缩 pipeline 为：① 扫描输入，把首次遇到的新序列按序加入字典（"XY"→1*、"YZ"→2*、"ZX"→3*）；② 后续输入若匹配字典序列则输出其索引（如 1*、2*）；③ 字典本身不写入压缩输出——解压器按相同顺序从压缩数据动态重建同一字典；④ 动态重建要求"短序列先于长序列被选中"，导致长序列只能压缩更晚的数据（LZ77 的 ⟨offset|length⟩ 正是为此设计）。注意 LZW 的字典是"按出现顺序隐式构建"的，与 RST 的"按 utility 显式挑选存储"形成对比。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- LZ 家族是 Deflate（LZ77+Huffman）、Zstandard（Zstd）、LZMA 等主流算法的"序列压缩"阶段（通常配合第二阶段的符号压缩）。硬件实现：IBM Power9/z15 数据压缩加速器（Deflate，ISCA'20）、CDPU（通用 LZ 家族加速器，支持 Deflate/LZ4/Zstd，ISCA'23）、TMCC 的 ASIC Deflate（面向内存压缩，MICRO'22）、OCP Project Zipline（开源 streaming Deflate RTL，存储/网络 I/O）。软件实现：zlib（软件 Deflate）。本论文应用场景：页级 LZ 被 Hyperscale Tiered Memory Expander Specification 强制用于硬件内存压缩，4KB 页整体作为单一压缩单元（压缩率随粒度增大而提高）。
涉及论文标题：
- Random-Access Hardware Sequence Compression
