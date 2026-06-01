# GPU的指令设计

ref:[[https://zhuanlan.zhihu.com/p/391238629](https://zhuanlan.zhihu.com/p/391238629)]

## 1、指令延迟

流水线的配置与ALU的latency有很大的关系。比如Volta前FFMA的延迟是6cycle，Volta及之后FFMA的延迟是4cycle，这绝对与流水线的改进有关。不过，这里的Latency并不是所有流水线的级数。因为Latency在程序中的表现形式是：**一个指令发射后，其结果需要多少周期才能就绪（也就是能被其他指令使用）。**

两个back-to-back dependent的ALU指令（比如FFMA R0, R1, R2, R0; FFMA R0, R3, R4, R0;），前一个FFMA只要在第二个FFMA读取操作数之前把结果写回GRF，那后一个FFMA就可以得到正确值。对应到上面的5级流水线形式，就是**前一个指令的WB要在后一个指令的ID前执行完就行（相当于4cycle延迟）**，最开始的IF那一级是不影响的。

CPU对于这种形式的依赖还有更激进的旁路逻辑（forwarding），可以直接在前一个ALU的EX后把结果直接送给后一个ALU的EX当输入，从而减少流水线的bubble，提高性能。

NV的GPU应该是没有这么紧凑的forwarding，但是**NV的operand collector可以作为一个公共的操作数中转站，理论上前一个ALU的结果写回到operand collector就可以被下一个ALU看到了，不一定要回到GRF。**当然，这个具体流水线的实现我也不是很清楚，有兴趣的同学可以尝试查查NV的专利。

## 2、指令吞吐

指令吞吐不仅与指令类型有关，还与微架构具体设计实现有关。它主要会受到以下一些因素的影响：

**功能单元的数目**

绝大多数指令的功能都需要专用或共享的硬件资源去实现，设计上配置的功能单元多，指令执行的吞吐才可能大。显然，只有最常用的那些指令，才能得到最充分的硬件资源。而为了节约面积，很多指令的功能单元会相互共享，所以他们的吞吐往往也会趋于一致。比如浮点的FFMA、FMUL都要用到一个至少24bit的整数乘法器（32bit浮点数有23bit尾数，小数点前还有1bit）。以前一些处理器有24bit的整数乘法指令，两者乘法器就可以共用，从而具有相同的吞吐（不过NV最近几代好像都没有这个指令，ptx以及内置函数的24bit乘法应该是多个指令模拟的）。而FADD虽然用不上那个乘法器，但可以与FFMA共用那个很宽的加法器，以及一些通用的浮点操作（特殊数的处理，subnormal flush之类）。32bit的整数乘法因为需要更宽的乘法器，有的就不会做成full throughput，甚至可能被拆分成多个指令（比如Maxwell和Pascal用三个16bit乘法指令XMAD完成一次32bit整数乘法）。Turing的IMAD应该是有意识的加宽了，所以32bit的IMAD与FFMA吞吐一样，但印象中带64bit加数的IMAD应该还是一半。再比如一些超越函数指令（MUFU类，比如rcp，rsq，sin，exp之类），由于实际使用量相对不会太频繁，多数是1/4的throughput。

**指令Dispatch Port和Dispatch Unit的吞吐**

一个warp的指令要发射，首先要eligible，也就是不要因为各种原因stall，比如指令cache miss，constant immediate的miss，scoreboard未就位，主动设置了stall count等等。

其次要被warp scheduler**选中**，由Dispatch Unit发送到相应的Dispatch Port(执行单元)上去。

Kepler、Maxwell和Pascal是一个Warp Scheduler有两个Dispatch Unit，所以每cycle最多可以发射两个指令，也就是双发射。而Turing、Ampere每个Warp Scheduler只有一个Dispatch Unit，没有双发射，那每个周期就最多只能发一个指令。

但是Kepler、Maxwell和Pascal都是一个Scheduler带32个单元（各种不同的unit），每周期都可以发新的warp。而Turing、Ampere是一个Scheduler带16个单元，每个指令要发两cycle。

最后要求Dispatch Port或其他资源不被占用，port被占的原因可能是**前一个指令的执行吞吐小于发射吞吐，导致要Dispatch多次**，比如Turing的两个FFMA至少要stall 2cycle，LDG之类的指令至少是4cycle。

**GPR读写吞吐**

绝大部分的指令都要涉及GPR的读写，由于Register File每个bank每个cycle的吞吐是有限的（一般是32bit），如果一个指令读取的GPR过多或是GPR之间有bank conflict，都会导致指令吞吐受影响。**GPR的吞吐设计是影响指令发射的重要原因之一，有的时候甚至占主导地位，功能单元的数目配置会根据它和指令集功能的设计来定。**

比如NV常用的配置是4个Bank，每个bank每个周期可以输出一个32bit的GPR。这样FFMA这种指令就是3输入1输出，在没有bank conflict的时候可以一个cycle读完。其他如DFMA、HFMA2指令也会根据实际的输入输出需求，进行功能单元的配置。

很多指令有**replay**的逻辑（参考Greg Smith在StackOverflow上的一个回答）。这就意味着有的指令一次发射可能不够。这并**不是之前提过的由于功能单元少而连续占用多轮dispath port，而是指令处理的逻辑上有需要分批或是多次处理的部分。**

比如constant memory做立即数时的cache miss，memory load时的地址分散，shared memory的bank conflict，atomic的地址conflict，甚至是普通的cache miss或是TLB的miss之类。根据上面Greg的介绍，Maxwell之前，这些replay都是在warp scheduler里做的，maxwell开始将它们下放到了各级功能单元，从而节约最上层的发射吞吐。不过，只要有replay，相应dispath port的占用应该是必然的，这样同类指令的总发射和执行吞吐自然也就会受影响。

## 3、指令执行特性

### 1、静态资源分配

GPU有一个很重要的设计逻辑是**尽量减少硬件需要动态判断的部分**。**GPU的每个线程和block运行所需的资源尽量在编译期就确定好，在每个block运行开始前就分配完成**（Block是GPU进行运行资源分配的单元，也是计算Occupancy的基础）。典型的运行资源有GPR和shared memory。**GPU程序运行过程中，一般也不会申请和释放内存（当然，现在有device runtime可以在kernel内malloc和free，供Dynamic Parallelism用，但这个不影响当前kernel能用的资源）**。

CPU在运行过程中有很多所需的资源是动态调度的。比如，x86由于继承了祖上编码的限制，ISA的GPR数目往往比物理GPR少，导致常常出现资源冲突造成假依赖。实际运行过程中，通常会有**register renaming将这些ISA GPR映射到不同的物理GPR**，从而减少依赖（有兴趣的同学可以研究下tomasulo算法）。

**GPU没有这种动态映射逻辑，每个线程的GPR将一一映射到物理GPR。**由于每个线程能用的GPR通常较多，加上编译器的指令调度优化，这种假依赖对性能的影响通常可以降到很低的程度。

每个block在运行前还会分配相应的shared memory，这也是静态的。这里需要明确的是，每个block的shared memory包括两部分，写kernel时固定长度的静态shared memory，以及启动kernel时才指定大小的动态shared memory。虽然这里也分动静态，但指的是编译期是否确定大小，在运行时总大小在kernel启动时已经确定了，kernel运行过程中是不能改变的。

其实block还有一些静态资源，比如用来做block同步的barrier，每个block最多可以有16个。我暂时没测试到barrier的数目对Occupancy的影响，也许每个block都可以用16个。另一种是Turing后才出现的warp内的标量寄存器Uniform Register，每个warp 63个+恒零的URZ。因为每个warp都可以分配到足额，应该对Occupancy也没有影响。另外每个线程有7个predicate，每个warp有7个Uniform predicate，这些也是足额，也不影响Occupancy。

GPU里还有一种半静态的stack资源，通常也可以认为是thread private memory或者叫local memory。多数情况下每个线程会用多少local memory也是确定的。不过，如果出现一些把local memory当stack使用的复杂递归操作，可能造成local memory的大小在编译期未知。这种情况编译器会报warning，但是也能运行。不过local memory有最大尺寸限制，当前是每个线程最多512KB（参考CUDA C Programming Guide, Table 15, Maximum amount of local memory per thread=512KB）。

### 2、顺序执行

**乱序执行是CPU**提高CPI的一个重要途径，但乱序执行无论是设计复杂度还是运行控制的开销都很大。CPU的乱序执行可以把一些不相关的任务提前（相关的也可以乱序，但要求顺序提交），从而提高指令并行度，降低延迟。

而GPU主要通过Warp切换的逻辑保持功能单元的吞吐处于高效利用状态，这样总体性能对单个warp内是否stall就不太敏感。

虽然GPU一般是顺序执行，但**指令之间不相互依赖的时候，可以连续发射而不用等待前一条指令完成。在理想的情况下，一个warp就可以把指令吞吐用满。**当然，实际程序还是会不可避免出现stall（比如branch），这时就需要靠TLP来隐藏这部分延迟。

### 3、显式解决依赖

既然是顺序执行，但同时又可以连续发射，那怎么保证不出现数据冒险呢？NV GPU现在主要有两类方式：

第一种是**固定latency**的指令，通过调节control codes中的stall count，或者插入其他无关指令(编译ordering)，保证下一条相关指令发射前其输入已经就位；

第二种是**不固定latency**的指令，就需要通过显式的设置和等待scoreboard来保证结果已经可用。

在x86的CPU中，memory结果的可见性是通过缓存的一致性来控制的，这样read-after-write之类的组合可以通过cache的可见性来保证，但多线程的情况也需要通过coherence和memory consistency model来保证。

GPU本身运行就是多线程的，同一个warp内也是通过scoreboard来保证次序。但多个warp之间，GPU也需要维护相应的coherence和memory consistency model，具体大家可以参考PTX文档: Memory Consistency Model。

当然这个逻辑虽然是这么设计的，估计偶尔也会有出bug的时候。Maxwell和之前的架构偶尔能看见编译器往程序内插一些NOP。大概就是硬件上有问题，靠编译器来强行修补。Turing上似乎已经比较少见了。

## 4、ILP和数据冒险

### 1、ILP的数据依赖

**ILP的逻辑主要是靠前一条指令不需要执行完成就能发射下一条无关指令，而TLP则是通过warp之间切换来隐藏延迟。**从另一个角度讲，ILP和TLP都可以增加可发射指令的数目，尽量减少功能单元的闲置，从而提高硬件利用效率。

ILP是线程内（更准确的说是Warp内）的并行逻辑，影响ILP的主要因素有两种，一是指令之间的依赖性，二是指令的资源竞争或冲突。依赖分显式和隐式。显式依赖主要是数据的相关性，隐式依赖则与资源竞争很相似，主要是两个指令都要使用某个特定含义的公共资源。

典型的**显式依赖**如：

`FFMA R3, R1, R2, R0;  // sm_75: stall 4 cycles
FFMA R6, R4, R5, R3;`
而**隐式的依赖**比如这种：

`// sm_61
1:  IADD   [RZ.CC](http://rz.cc/), R0, R1 ;   // set condition code as carry
2:  IADD.X R2, RZ, R2 ;      // use condition code as carry
3:  IADD   [RZ.CC](http://rz.cc/), R3, R4 ;
4:  IADD.X R5, RZ, R5 ;`
IADD可以把进位存到专门的CC寄存器（类似x86的carry flag），然后IADD.X可以把这个CC寄存器当成carry读进来再做加和。这里指令1和3存在RZ.CC的WAW冒险,不能将2和3交换来隐藏1-2之间和3-4之间的stall。

而在Turing里这种指令已经可以显式的用Predicate来存储carry，如：

`// sm_75
1:  IADD3   R4, P0, R0, R2, RZ ;         // set P0 to carry out
2:  IADD3   R10, P1, R6, R8, RZ ;        // set P1 to carry out
3:  IADD3.X R5, R1, R3, RZ, P0, !PT ;    // use P0 as carry in
4:  IADD3.X R11, R7, R9, RZ, P1, !PT ;   // use P1 as carry in`
这样原本的1-3和2-4的顺序可以排列成1-2-3-4的顺序，正确性互不影响，从而相互隐藏延迟，提高ILP。

注：我其实没明白IADD3.X的第二个Predicate（!PT）是干什么用的，SASS有很多指令都会带一个predicate输入，多数没看到明显的价值。

这种相对更独立的指令集设计其实有点类似函数式编程：操作专门carry寄存器可以看做是stateful的操作，改成可编程的Predicate后就成为只与输入输出有关的stateless操作，不改变机器状态。也可以认为是所有需要用carry寄存器做输入输出的指令，都需要被序列化。通过与当前机器状态解耦，获得更大的指令调度自由度，**编译器的后端优化也会更加方便。**

### 2、显式操作指令、运行时操作(隐式)的设计权衡

NV GPU在很早就开始**有意识的淘汰这种含隐式输入输出的指令**。比如早期使用隐式栈的call、ret、break等等（当然其实这和ILP关系已经不太大了）。例如SM61中要return时需要先显式的用PRET设置某个隐式的调用栈，然后直接用RET返回。显然在RET时这个栈必须是对应当初PRET设置的值（中间能不能再进出栈没仔细研究），否则就会出错。而Turing直接使用带GPR地址的指令进行操作，就消除了这种隐式栈的操作过程，减少了指令之间复杂依赖对编译器的干扰。

`// RET in sm_61
PRET 0x258 ;
...
RET;`

`// RET.REL in sm_75
MOV R20, 32@lo((*Z7argtestPiS_S* + .L_9@srel)) ;   // relocation with addend
MOV R21, 32@hi((*Z7argtestPiS_S* + .L_9@srel)) ;
...
RET.REL.NODEC R20 `(*Z7argtestPiS_S*);`

`// RET.ABS in sm_75
RET.ABS R32 `(*Z7argtestPiS_S*);`

再稍微扩展一点。x86中有control register来控制如何做浮点数的rounding，是否做subnormal的flush（x87 FPU control register控制普通的FPU指令，MXCSR控制SSE指令）。但在SASS中，FFMA、FMUL、DFMA等浮点运算指令，每个指令都可以自主控制是否打开flush（使用FTZ modifier），如何做rounding（RM, RP, RZ, RN）。这就意味着每个指令可以自主决定当前指令的运行方式，而不用改变机器状态。不同模式混用时，就不需要保存和恢复control register了。

那再再扩展一点，x86其实也有控制FP exception的寄存器，NV GPU里是怎么操作的呢？我好像没看见，感觉是被去掉了。这还是一个挺值得思考的问题~

当然，**也不是说所有的隐式都应该变成显式**。比如每个Warp都有一个隐式的active mask，用来标记当前warp中divergence的情况。active mask与指令predicate的“AND”会共同决定当前指令是否起作用。

那把mask交由指令显式操作有意义吗？我觉得没有，因为这没有太多额外的可编程价值。首先divergence造成的mask变化只能被分支或分支同步指令修改，其他指令需要控制效果直接用predicate就可以了，没必要操作mask。其次，这个操作本身是非常固化的，增加相关操作指令并不会带来新功能，反而会增加指令负担（相当于每次可能有divergence的分支时，都要显式的保存和设置mask，遍历不同divergence的分支时就更麻烦了）。

因此，在有predicate的情况下，**active mask还是做成隐式的比较合理。当然，volta后的Independent Thread Scheduling也要自主控制和依赖内部的mask状态，就更没法做成显式的了。不过，虽然mask是隐式的，但divergence后重新converge一般是显式的（通过BSSY和BSYNC指令），否则程序就不知道应该在哪个点join了。**