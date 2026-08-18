## Intel DSA Transparent Offload（DTO）库

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DTO（DSA Transparent Offload）是 Intel 的用户态共享库，让未修改的应用透明地使用 DSA：拦截标准 C 内存函数 memcpy/memmove/memset/memcmp，把超过阈值的调用转换为 DSA Memory Move/Fill/Compare 操作并异步等待完成。应用可通过 `-ldto` 直接链接，或经 LD_PRELOAD 注入而无须重编译。DarkStream 用它把 Chromium 浏览器与 DL 推理的内存操作卸载到 DSA（memset→Fill、memcpy/memmove→Memory Move），构成侧信道受害端的 DSA 流量。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
运行期拦截流程（非编译期变换，最接近"框架级在线算子卸载"）：应用调用 memcpy(dst,src,n) → DTO 拦截函数检查 n ≥ DTO_MIN_BYTES（默认 16384；DarkStream 用最低推荐设置卸载 >8 KB 的操作）→ 构造 DSA descriptor（源/目标地址、长度、操作码）→ 经用户态 portal（ENQCMD/MOVDIR64B）提交到配置的 WQ（DTO_WQ_LIST 指定 /dev/dsa/*）→ 按 DTO_WAIT_METHOD（yield/busypoll/umwait）等待完成记录 → 返回。DTO_AUTO_ADJUST_KNOBS 启发式自适应卸载阈值，DTO_IS_NUMA_AWARE 做 NUMA 感知，DTO_COLLECT_STATS 收集直方图统计，DTO_USESTDC_CALLS 在 DSA 页错误时回退标准 C 库。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
源码与用法：https://github.com/intel/DTO（dto.c 实现、dto-test.c 多线程样例、test.sh 展示 -ldto 与 LD_PRELOAD 两种用法、dto-4-dsa.conf 示例配置）；`LD_PRELOAD=libdto.so ./app` 或 `gcc app.c -ldto`。适用场景：数据搬移密集且对象较大的既有应用免改造加速（存储、网络、浏览器、推理运行时）；小对象卸载因 DSA 调用延迟反而劣化，故有最小阈值与自适应调参。DarkStream 中的用法：`make libdto` 构建后以最小设置（8 KB）卸载 Chromium/推理内存操作，使受害者负载在共享 DSA 上产生可被攻击者观测的争用流量。层次归类说明：DTO 是运行时拦截库而非编译器，归入编译框架为最接近层次（框架内部组成 + 在线算子卸载优化）。

涉及论文标题：
- DarkStream: Exploiting Internal Throughput Contention in Data Streaming Accelerator for Timing Attacks
