## Compact Storage of Low-Precision Data (低精度数据紧凑存储)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
低精度数据紧凑存储（Compact Storage）是将bit width小于8的元素连续打包进字节序列的存储方式，消除字节内的bit gaps。例如4个int6元素（共24 bits）紧凑存储在3个uint8字节中，而非各浪费2 bits独立存储。由于单个低精度值可能跨越两个连续字节边界（如Figure 8中的b[1]），访问需要bitwise操作：提取用AND+SHIFT+OR组合，写入用MASK+OR。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
紧凑存储下的元素访问伪代码：
```
# 读取第i个bit_width位宽的紧凑存储元素
def read_compact(u8* data, int i, int bit_width):
    bit_offset = i * bit_width
    byte_offset = bit_offset // 8
    bit_pos = bit_offset % 8
    val = data[byte_offset] >> bit_pos
    if bit_pos + bit_width > 8:  # 跨字节边界
        val |= data[byte_offset+1] << (8 - bit_pos)
    return val & ((1 << bit_width) - 1)
```

在Tilus中，紧凑存储是低精度weight loading的基础：权重tensor被变换为连续u8字节序列后，通过LoadGlobal以标准类型加载到registers，再通过View指令零开销reinterpret回原始低精度类型（含正确的layout），无需在加载时逐元素做bitwise提取。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Tilus在kernel启动前运行预处理kernel（transform_b, Figure 9），将权重从原始类型（如i6[K,N]）变换为紧凑u8格式（u8[K/BK, N/BN, BK*BN*6/8]）。变换后的tensor中每BK×BN tile的所有bits连续排列，实现coalesced memory access。compact storage方法generalize到任意bit width：给定per-thread bytes数n和T线程，使用u8 dtype和layout local(n/gcd(n,16)).spatial(T).local(gcd(n,16))进行加载。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation
