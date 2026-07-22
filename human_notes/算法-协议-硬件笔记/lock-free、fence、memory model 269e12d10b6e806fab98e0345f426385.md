# lock-free、fence、memory model

[https://zhuanlan.zhihu.com/p/342921323](https://zhuanlan.zhihu.com/p/342921323)

# lock、lock-free、wait-free

## 概念

[https://blog.csdn.net/wxj1992/article/details/103649056](https://blog.csdn.net/wxj1992/article/details/103649056)

**lock**可能出现的情况：某时刻，获得锁的线程可能被调度、中断、长时间IO而**不释放锁**，导致其余等待锁的**线程全部阻塞且时间不可预测**；

**lock-free**：要求**任意时刻存在线程执行**而不允许全部阻塞，即算法全局**整体推进**，但由于线程串行使用临界区，**存在线程的完成时间不可预期**；

[https://noahyzhang.github.io/2023/10/11/项目/9天带你走进无锁化编程/5.研读C++语言boost库中无锁队列的实现/](https://noahyzhang.github.io/2023/10/11/%E9%A1%B9%E7%9B%AE/9%E5%A4%A9%E5%B8%A6%E4%BD%A0%E8%B5%B0%E8%BF%9B%E6%97%A0%E9%94%81%E5%8C%96%E7%BC%96%E7%A8%8B/5.%E7%A0%94%E8%AF%BBC++%E8%AF%AD%E8%A8%80boost%E5%BA%93%E4%B8%AD%E6%97%A0%E9%94%81%E9%98%9F%E5%88%97%E7%9A%84%E5%AE%9E%E7%8E%B0/)

**wait-free**：要求**每个线程都有确定的完成时间或上界**，不确定的串行调度会导致线程完成时间不可估计，因此每个slice所有线程都要有进展；

> **[图片提取文字 (image.png)]:**
> 的临界区,比如最常用的一种数据结构:并发队列,生产者和消费者需要访问队列的公共内存进行写和读。目前对于临界区保护通常可以分为三个级别:互斥、lock-free和wait-free。
> 
> 在正式介绍memory model<sup>Q</sup>之间,我想先聊聊多核并发编程中最核心的技术:临界区保护,利用多线程做并发的任务中通常都会有公共
> 
> ## 2.1 互斥
> 
> 最简单大家也最熟悉的的临界区保护技术自然是互斥,每个线程访问之前都需要获得互斥锁,如果被别的线程占用了就阻塞等待。这是一种典型的悲观锁算法,很明显,当进入临界区的线程发生阻塞,或被操作系统换出时,会出现全局阻塞,因为获得锁的线程被换出无法执行操作,而未获得锁的线程也只能一同等待,出现了阻塞传播,如果另一个线程先进入临界区,有可能反而可以更快顺利完成。因为存在
> 
> 全局阻塞的可能性,采用互斥技术进行临界区保护的算法有着最低的阻塞容忍能力。
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/24358451-ce73-4f42-b1fb-a75cc46a7fde.png)

> **[图片提取文字 (image.png)]:**
> ## 2.2 lockfree
> 
> 程序员总是追求极致的,在各路大神的引领下,针对互斥的阻塞问题,,Lock free的概念应运而生并逐渐流行起来。Lock free programing,字面意思就是无锁编程,很多人的理解是成没有用到各类显式锁的编程,这个理解并不准确,其实主要是取自非阻塞算法等级中的一种分类术语,本质上是一种乐观锁算法。这个概念有着不同的表述方式,这里先贴一个preshing大佬博客里通俗易懂的图:
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image.png)

> **[图片提取文字 (image.png)]:**
> ## 维基百科上的定义如下:
> 
> Lock free允许单独的线程个体阻塞,但是会保证系统整体上的吞吐,如果一个算法对应的程序的线程在运行了足够长时间的情况下,至少有一个线程取得了进展,那么我们说这个算法是lock-free的。
> 
> 概括起来就是,如果涉及到共享内存的多线程代码在多线程执行下不可能互相影响导致被hang住,不管OS如何调度线程,至少有一个线程在做有用的事,那么就是lock-free。前面讲到了使用了锁的代码肯定不是lock free,因为一个线程加锁后如果被系统切出去了其他所有线程都处于等待中。但是没用锁也不一定是lock free,因为普通的代码逻辑也可能会导致一个线程hang住另一个线程。锁之所以在高并发的时候表现很差,主要原因就是加锁的线程会hang住其他待加锁的线程,lock-free可以很好的解决这一问题。
> 
> 具体到实现上,本质上就是首先假设临界区不存在竞争,各个线程直接开始临界区的执行,但是通过良好的设计,让这段预先的执行是无冲突可回滚的。最终有一个需要同步的提交操作,一般基于原子变量CAS,或者版本校验等机制完成。在提交阶段如果发生冲突,那么被仲裁为失败的各方需要对临界区预执行进行回滚,并重新发起一轮尝试。
> 
> 需要强调的一点是,并不是说lock-free的算法就一定比加锁的算法好,lock-free需要处理更多更复杂的race condition和ABA等问题,编写出合理的lock-free代码也需要更深厚的功底,需要对底层有更多地了解,完成相同目的的代码会比用锁更复杂,执行时间可能更长,代码也更难理解。很多场景合理地使用锁就能很好的胜任,lock-free和锁之间在应用场景上更多的是一种互补的关系。lock-free算法的价值在于其保证了一个或所有线程始终在做有用的事,而不是绝对的高性能。但lock-free相较于锁在并发度高(竞争激烈导致上下文切换开销变得突出)的某些场景下会有很大的性能优势,比如实现一个多线程的lock-free queue,总的来说,在多核环境下,lock-free是很有意义的。
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%201.png)

> **[图片提取文字 (image.png)]:**
> ## 2.3 wait-free
> 
> 要区别也就体现在这个吞吐的问题上,在无全局停顿的基础上,lock-free进一步保障了任意算法参与线程,都应该在有限的步骤内完成。 不只是整体算法时时刻刻存在有效计算,每个线程视角依然是需要持续进行有效计算。这就要求了多线程在临界区内不能被细粒度地串行 起来,而必须是同时都能进行有效计算。虽然理论角度存在不少有Wait Free级别的算法,但大多并不具备工业使用价值。主要是由于 wait-free限制了同时有进展,但是并没有描述这个进展有多快。因此进一步又提出了细分子类,以比较有实际意义的Wait-Free Population Oblivious级别来说,额外限制了每个参与线程必须要在预先可给出的明确执行周期内完成,且这个周期不能和与参与线程数相 关。这一点明确拒绝了一些类似线程间协作的方案(这些方案往往引起较大的缓存竞争),以及一些需要很长很长的有限步来完成的设 计。时至今日各种数据结构上工业可用的wait-free算法依旧是一项持续探索中的领域。
> 
> lock-free技术主要解决了临界区内的阻塞传播问题,但是本质上,依然是多个线程排队顺序经过临界区。而wait-free级别和lock-free的主
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%202.png)

## 例子

**有锁**算法在某个slip中make no progress；

> **[图片提取文字 (image.png)]:**
> MakeProgress
> 
> MakeNoProgress
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2031.png)

> **[图片提取文字 (image.png)]:**
> 假设一个程序如图中一共有T1、T2和T3三条线程,按照时间顺序运行,Slip表示某个理论上的步骤,不一定是时间片,三条线程可能同时运行在多个CPU核上,这样从时间上看是并行执行的,也可能是在一个CPU核<sup>+</sup>上并发执行的,不过这不影响我们对上图的理解。
> 
> 可以看到这种模式下某些线程存在某些Slip (T2-Slip1、T3-Slip2等)没有取得任何的Progress,甚至在同一个Slip(图中 的Slip3)内三条线程均没有任何的Progress。这一般就是阻 塞算法带来的效果。举个可能的例子,假设T1、T2和T3需要 申请一把锁访问共享数据,Slip2-T2申请对mutex<sup>†</sup>加锁,然 后T2时间片用完让出CPU,然后进入Slip3,T1和T3均因无法 对mutex加锁导致无法Make Progress, T2由于某些原因(高 优线程抢占、IO阻塞等)也没有Make Progress,此时就出现 程序整体没有Make Progress。我们很容易分析出,这实际上 就是因为对共享数据的访问互斥造成的,这也是我们想要做并 发共享数据结构<sup>\*</sup>的原因。
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%203.png)

**lock-free**要求算法**总存在**一个线程make progress；

> **[图片提取文字 (image.png)]:**
> An algorithm is lock-free if, when the program threads are run for a sufficiently long time, at least one of the threads makes progress (for some sensible definition of progress).
> 
> Lock-free被定义为程序运行充分长一段时间,至少有一个线程可以Make Progress,这和本文的解释显然是一致的。
> 
> 最后,对Wait-free和Lock-free的理解只需要记忆本文中三张 图的特点,即可对一个算法进行准确有效的判断。
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%204.png)

**wait-free**要求算法每个操作/线程有执行的**上界**；

> **[图片提取文字 (image.png)]:**
> An algorithm is wait-free if every operation has a bound on the number of steps the algorithm will take before the operation completes.
> 
> Wait-free被定义为算法在完成最终目标之前,每一个操作都 能在有限步内实现。这里的第一个operation可以理解为本文 的Make Progress, step可以理解为Slip, 当线程的每一个 Slip都在Make Progress时,那么针对一个特定的算法就一定 能在有限个Slip内完成,这是显而易见的。
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%205.png)

> **[图片提取文字 (image.png)]:**
> MakeProgress MakeNoProgress
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2033.png)

> **[图片提取文字 (image.png)]:**
> 可以看到,程序任何一个Slip都至少有一个线程可以Make Progress,但是还是存在某些线程的Slip (T2-Slip1、T3-Slip3等) 没有Make Progress, 此时的定义就是Lock-free。 Lock-free给出了比普通加锁实现更高的保障,即程序在任意 时间内至少有一条线程能Make Progress。
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%206.png)

> **[图片提取文字 (image.png)]:**
> T1 T2 T3
> 
> MakeProgress
> 
> MakeNoProgress
> 
> ![](_page_0_Picture_3.jpeg)
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%207.png)

> **[图片提取文字 (image.png)]:**
> ## 在任意Slip内,每一条线程都能Make Progress。
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%208.png)

# fence

[https://zhuanlan.zhihu.com/p/41872203](https://zhuanlan.zhihu.com/p/41872203)

> **[图片提取文字 (image.png)]:**
> 在编写single writer lock-free代码的时候,通常需要手动使 用memory fence/barrier来确保修改对其他core可见并防止 乱序(对于multiple writer的情况一般需要atomic RMW操作 <sup>†</sup>,隐含了memory fence,不需手动加)。一般来说 memory fence分为两层: compiler fence和CPU fence, 前 者只在编译期生效,目的是防止compiler生成乱序的内存访问 指令;后者通过插入或修改特定的CPU指令,在运行时防止内 存访问指令乱序执行。下面分别说下在X86/GCC<sup>†</sup>环境下我对 这两种memory fence用法的一些经验。
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%209.png)

## compiler fence：约束编译器的reorder

> **[图片提取文字 (image.png)]:**
> GCC的compiler fence有一个众所周知的写法:
> 
> asm volatile("": : :"memory")
> 
> 那么这句话是什么意思呢?它只是插入了一个空指令"",什么 也没做。其实不然,这句话的关键在最后的"memory" clobber, 它告诉编译器: 这条指令(其实是空的)可能会读 取任何内存地址,也可能会改写任何内存地址。那么编译器会 变得保守起来,它会防止这条fence命令上方的内存访问操作 移到下方,同时防止下方的操作移到上面,也就是防止了乱 序,是我们想要的结果。
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%2010.png)

> **[图片提取文字 (image.png)]:**
> 但这还没完,这条命令还有另外一个副作用:它会让编译器把 所有缓存在寄存器中的内存变量flush到内存中,然后重新从 内存中读取这些值。这并不一定是我们想要的结果,比如有些 变量只在当前线程中使用,留在寄存器中很好,多了一对写/ 读内存操作是不必要的开销。
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%2011.png)

> **[图片提取文字 (image.png)]:**
> 那么有没有办法避免这种副作用呢?我们可以通过gcc内联汇编命令的input和output操作符明确指定哪些内存操作不能乱序,如这个例子:
> 
> ```
> WRITE(x)
> asm volatile("": "=m"(y) : "m"(x):) // memory fence
> READ(y)
> ```
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%2012.png)

> **[图片提取文字 (image.png)]:**
> 这里先对变量x进行写操作,后对变量y进行读操作,中间的内 联汇编告诉编译器插入一条指令(其实是空的),它可能会读 x的内存,会写y的内存,因此编译器不会把这两个操作乱序。 这种明确的memory fence的好处是: 使编译器尽量少的对其 他不相关的变量造成影响,避免了额外开销。
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%2013.png)

https://github.com/MengRao/SPSC_Queue/blob/master/SPSCQueue.h

## CPU fence：约束CPU的reorder

> **[图片提取文字 (image.png)]:**
> X86属于strong memory model,这意味着在大多数情况下 cpu会保证内存访问指令有序执行。具体的说,如果对内存读 (Load)和写(Store)操作进行两两组合: LoadLoad, LoadStore, **StoreLoad**, StoreStore, 只有StoreLoad组合可 能乱序,而且Store和Load的内存地址必须是不一样的。在上 面的队列模板库的例子中,由于只使用了StoreStore(对应 C++11的Release memory order)和LoadLoad(对应C++11
> 
> 的Acquire memory order),因此不需要额外的CPU fence。
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ## 对于可能出问题的StoreLoad操作,有如下例子说明:
> 
> | Processor 0         | Processor 1        |
> |---------------------|--------------------|
> | mov [ _x], 1 // M1  | mov [ _y], 1 // M3 |
> | mov r1, [ _y] // M2 | mov r2, [_x] // M4 |
> 
> Initially x == y == 0
> 
> r1 == 0 and r2 == 0 is allowed
> 
> Table 2.3.a: Loads may be reordered with older stores
> 
> 有两个变量x = 0, y = 0, 两个cpu分别把x, y设1, 并读取y和x的值,如果不存在乱序的话我们期望读取的结果不会全为0。不幸的是,全为0的情况可能发生。我写了个测试这种情况的代码:
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%2015.png)

```cpp
volatile int g_cnt = 0;
void threadfun(volatile int& loop_cnt, volatile int& x, volatile int& y, volatile int& r) {
    while(true) {
        while(g_cnt == loop_cnt) ;

        asm volatile("" ::: "memory");

        nocpufence(x, y, r);

        asm volatile("" ::: "memory");
        loop_cnt++;
    }
}

int main() {
    alignas(64) volatile int cnt1 = 0;
    alignas(64) volatile int cnt2 = 0;
    alignas(64) volatile int x = 0;
    alignas(64) volatile int y = 0;
    alignas(64) volatile int r1 = 0;
    alignas(64) volatile int r2 = 0;
    thread thr1(threadfun, ref(cnt1), ref(x), ref(y), ref(r1));
    thread thr2(threadfun, ref(cnt2), ref(y), ref(x), ref(r2));

    int detected = 0;
    while(true) {
        x = y = 0;
        asm volatile("" ::: "memory");
        g_cnt++;
        while(cnt1 != g_cnt || cnt2 != g_cnt)
            ;

        asm volatile("" ::: "memory");
        if(r1 == 0 && r2 == 0) {
            detected++;
            cout << "bad, g_cnt: " << g_cnt << " detected: " << detected << endl;
        }
    }
    return 0;
}
```

> **[图片提取文字 (image.png)]:**
> ## 运行结果:
> 
> ```
> bad, g cnt: 1272158 detected: 19791
> bad, g cnt: 1273128 detected: 19792
> bad, g cnt: 1273348 detected: 19793
> bad, g cnt: 1273985 detected: 19794
> bad, g cnt: 1274933 detected: 19795
> ^(
> ```
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%2016.png)

> **[图片提取文字 (image.png)]:**
> 为了防止这种CPU乱序,我们需要添加CPU memory fence。X86专门的memory fence指令是"mfence";另外还可以使用"lock add"指令起到相同的效果,后者开销更小,kernel的smp\_mb宏就是用"lock add"实现。
> 
> 另外,我们还可以把Store和memory fence组合在一起,用一个指令实现: "xchg", kernel的smp\_store\_mb宏就是这样实现的。
> 
> 在上面的测试代码中,用mfence/lockadd/xchg函数替换 nocpufence即可解决问题。
![image.png](lock-free%E3%80%81fence%E3%80%81memory%20model/image%2017.png)

```cpp
#include <bits/stdc++.h>
using namespace std;

inline void nocpufence(volatile int& x, volatile int& y, volatile int& r) {
    x = 1;
    asm volatile("" ::: "memory");
    r = y;
}

inline void mfence(volatile int& x, volatile int& y, volatile int& r) {
    x = 1;
    asm volatile("mfence" ::: "memory");
    r = y;
}

inline void lockadd(volatile int& x, volatile int& y, volatile int& r) {
    x = 1;
    asm volatile("lock; addl $0,0(%%rsp)" ::: "memory", "cc");
    r = y;
}

inline void xchg(volatile int& x, volatile int& y, volatile int& r) {
    int tmp = 1;
    asm volatile("xchgl %0, %1" : "+r"(tmp), "+m"(x)::"memory", "cc"); // swap(x, tmp)
    r = y;
}
```

# memory model/order

[https://blog.csdn.net/wxj1992/article/details/103649056](https://blog.csdn.net/wxj1992/article/details/103649056)

**lock显式规定**了指令同步的约束，编译器和CPU以lock为依据重排指令，多线程执行不会出错；

**memory order**是**lock-free**程序对内存访问指令次序的约束，以便编译器和CPU正确重排指令；

编程者、编译器、硬件对memory的共识是**memory model**，主要体现在**memory order**；

[https://blog.csdn.net/wxj1992/article/details/103917093](https://blog.csdn.net/wxj1992/article/details/103917093)

# memory order的标准定义

[https://blog.csdn.net/wxj1992/article/details/103656486](https://blog.csdn.net/wxj1992/article/details/103656486)

# 原子变量定义memory order

[https://blog.csdn.net/wxj1992/article/details/103843971](https://blog.csdn.net/wxj1992/article/details/103843971)

# atomic threads fence定义memory order

[https://blog.csdn.net/wxj1992/article/details/103917093](https://blog.csdn.net/wxj1992/article/details/103917093)

# memory order在编译器和CPU的作用

[https://blog.csdn.net/wxj1992/article/details/104266983](https://blog.csdn.net/wxj1992/article/details/104266983)