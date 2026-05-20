2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO) 

# BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration 

Yuzong Chen _Cornell University_ New York, NY, USA yc2367@cornell.edu 

Jian Meng _Cornell University_ New York, NY, USA jm2787@cornell.edu 

Jae-sun Seo Mohamed S. Abdelfattah _Cornell University Cornell University_ New York, NY, USA New York, NY, USA js3528@cornell.edu mohamed@cornell.edu 

_**Abstract**_ **—Bit-level sparsity methods skip ineffectual zero-bit operations and are typically applicable within bit-serial deep learning accelerators. This type of sparsity at the bit-level is especially interesting because it is both orthogonal and compatible with other deep neural network (DNN) efficiency methods such as quantization and pruning. Furthermore, it comes at little or no accuracy degradation and can be performed completely post-training. However, current bit-sparsity approaches lack practicality because of (1) load imbalance from the random distribution of zero bits, (2) unoptimized external memory access because all bits are fetched from off-chip memory, and (3) high hardware implementation overhead, including large multiplexers and shifters to support sparsity at the bit level.** 

**In this work, we improve the practicality and efficiency of bitlevel sparsity through a novel algorithmic bit-pruning, averaging, and compression method, and a co-designed efficient bit-serial hardware accelerator. On the algorithmic side, we introduce bidirectional bit sparsity (BBS). The key insight of BBS is that we can leverage bit sparsity in a symmetrical way to prune either zero-bits or one-bits. This significantly improves the load balance of bit-serial computing and guarantees the level of sparsity to be more than 50%. On top of BBS, we further propose two bit-level binary pruning methods that require no retraining, and can be seamlessly applied to quantized DNNs. Combining binary pruning with a new tensor encoding scheme, BBS can both skip computation and reduce the memory footprint associated with bi-directional sparse bit columns. On the hardware side, we demonstrate the potential of BBS through** _**BitVert**_ **, a bitserial architecture with an efficient PE design to accelerate DNNs with low overhead, exploiting our proposed binary pruning. Evaluation on seven representative DNN models shows that our approach achieves: (1) on average** 1 _._ 66 _×_ **reduction in model size with negligible accuracy loss of** _<_ 0 _._ 5% **; (2) up to** 3 _._ 03 _×_ **speedup and** 2 _._ 44 _×_ **energy saving compared to prior DNN accelerators.** _**Index Terms**_ **—Deep learning accelerator, bit-serial computing, hardware-software co-design, sparsity, model compression** 

## I. INTRODUCTION 

Deep neural networks (DNNs) have demonstrated remarkable accomplishments in many important fields such as computer vision and natural language processing. However, the growth of DNN model size and complexity continues to outpace the scaling of compute performance in existing hardware platforms [12]. Bridging this performance gap is very desirable for wider adoption of DNNs, particularly in edge scenarios that demand both high performance and energy efficiency. Codesigning novel DNN compression algorithms, together with accelerators for the efficient deployment of the compressed models, is a promising way to achieve this goal. 

Numerous efficiency algorithms [21], [30], [31] and hardware prototypes [6], [13], [14], [42], [43] have been proposed to leverage _value-based sparsity_ in DNNs to reduce the cost of storing and deploying DNNs. Yet the degree of such value sparsity, which depends on the underlying model architecture, can strongly limit the resulting hardware performance. For instance, recent transformer-based DNNs show limited or no activation sparsity with GeLU and sigmoid activation functions [7], [9]. Even for single-sided sparse accelerators that target weight sparsity, plenty of time and cost are spent on retraining the model to balance the degree of sparsity and accuracy loss. Unfortunately, in many real-world cases, retraining may become impractical for end users due to cost constraints and lack of access to the original training dataset [3], [39]. This challenge is particularly pronounced in recent large language models [40], [47] that contain billions of parameters, making retraining even more resource and data intensive. Hence, there is a strong need to further enhance the efficiency of DNN accelerators _without imposing retraining_ . 

Another line of DNN compression research focuses on _posttraining quantization_ (PTQ), which represents DNN operands in lower precision without retraining the model [15], [24], [25], [32], [36], [44], [45]. For example, researchers have designed new quantization data types such as the Microscaling format [36], where a group of low-precision operands can share an 8-bit exponent to balance the accuracy and memory footprint. However, Microscaling still requires a floating-point pipeline to handle the shared 8-bit exponent, resulting in higher hardware cost than integer quantization. On the other hand, state-of-the-art PTQ algorithms can already reduce the operand precision to 8-bit integer with negligible accuracy loss [24], [32], [44]. Unfortunately, a quantized 8-bit DNN shows extremely low value sparsity (less than 5% as will be shown in the next section), since it tries to utilize all quantization levels as much as possible to reduce the quantization error. This fundamental quantization-sparsity tension poses a big performance bottleneck in existing value-based DNN accelerators [16], [38]. 

In order to jointly exploit the efficiency of quantization and sparsity, a series of bit-serial DNN accelerators exploit _bitlevel sparsity_ [1], [19], [20], [26], [37], [39]. Unlike coarsegrained value sparsity that is incompatible with quantization, the bit-level sparsity targets the abundant _zero bits_ in the 

979-8-3503-5057-9/24/$31.00 ©2024 IEEE 551 DOI 10.1109/MICRO61859.2024.00048 

**==> picture [234 x 168] intentionally omitted <==**

**----- Start of picture text -----**<br>
Sign Magnitude w. Zero   2's Complement w.<br>Pruned Bit Columns ❶ Bit Columns (Previous)   ❶ Bi-directional Sparse<br>Original 2's Complement MSE = 1.75 Columns (Ours), MSE = 0.75<br>77 0 1 0 0 1 1 0 1 76 0 1 0 0 1 1 0 0 78 0 1 0 0 1 1 1 0<br>-25 1 1 1 0 0 1 1 1 -24 1 0 0 1 1 0 0 0 -26 1 1 1 0 0 1 1 0<br>6 0 0 0 0 0 1 1 0 4 0 0 0 0 0 1 0 0 6 0 0 0 0 0 1 1 0<br>-11 1 1 1 1 0 1 0 1 -12 1 0 0 0 1 1 0 0 -10 1 1 1 1 0 1 1 0<br>PTQ INT5 Weight  ❷ KL Div = 1.8 E-6 ❷ KL Div = 8 E-7<br>KL Div = 5.1 E-6<br>ResNet-50<br>Conv4.1.3<br>INT8 Weight<br>(a)  (b) (c)<br>**----- End of picture text -----**<br>


Fig. 1: Comparison of different model compression approaches. (a) Example of a 4-value group and the weight distribution of a ResNet-50 layer before and after PTQ. (b) 1 Bit-sparsity enhancement by generating three zero bit columns using sign-magnitude format, 2 achieving lower KL divergence than PTQ but still losing many quantization levels. (c) 1 BBS generates three bidirectional sparse bit columns and is able to preserve all quantization levels of 8-bit precision, 2 leading to much lower KL divergence. 

binary representation of operands, thus is both compatible and orthogonal to other forms of DNN redundancy. Stripes [19] is an early bit-serial prototype that uses reduced precision for DNNs to scale the performance. Pragmatic [1], Laconic [37] and Bitlet [26] propose to skip zero-bit operations from different perspectives. However, the distribution of zero bits is generally random, whether in an individual operand or a group of operands, leading to significant workload imbalance. A direct consequence is that these accelerators must still fetch all data bits from off-chip memory, and use sophisticated hardware schedulers to skip zero-bit operations as much as possible during on-chip computation. The latter usually incurs non-trivial hardware overhead. 

To reduce both memory access and scheduling overhead of bit-serial computing, BitWave [39] employs a bit-columnserial approach, which examines the sparsity of the same bit significance across a group of operands. If a bit column contains all zero-bits, then it does not need to be stored in memory. Moreover, BitWave proposes a bit-sparsity-enhancing technique based on sign-magnitude formatted weights to selectively flip bits to zero. With this _bit-flip_ technique, BitWave is able to further compress a quantized 8-bit DNN by generating more zero bit columns. As a result, it has demonstrated the potential to achieve higher performance than other bit-serial accelerators [1], [19], [26]. 

Despite these approaches exploring bit sparsity at varying degrees, they still suffer from one significant drawback: bit sparsity is only limited to zero bits. To demonstrate this problem, consider Figure 1(a) that shows a group of four INT8 values, as well as the INT8 weight distribution of a layer in ResNet-50. If we want to further reduce the bit-width to, _e.g._ , 5-bit, conventional PTQ needs coarse-grained clipping and re-scaling so that the quantization mean square error (MSE) is minimized. Nevertheless, no matter what PTQ algorithm 

is used, the resulting distribution can only have 2[5] = 32 discrete quantization levels, resulting in large KL divergence, a common metric to quantify the difference between two distributions [17]. On the other hand, previous bit-sparsityaware works [23], [35], [39] leverage sign-magnitude format to prune bit columns at the group level as shown in Fig. 1(b). Given that DNN weights are typically small, many inherent zero bit columns exist ( _e.g._ , the third bit columns in Fig. 1(b)), leading to less sparse columns enforced ( _e.g._ , the seventh and eighth bit columns in Fig. 1(b)) to achieve the effective 5-bit data width. As a result, they can preserve more quantization levels and achieve lower KL divergence and better accuracy than PTQ. However, if there is no inherent sparse bit column in a group, all lower significant bit columns must be flipped to zero, leading to reduced quantization levels especially in intervals with large absolute values ( _e.g._ , _> |_ 50 _|_ in Fig. 1(b)). 

**Our focus:** this work proposes a novel sparsity concept called _bi-directional bit-level sparsity_ (BBS) and the associate bit-serial accelerator design named _BitVert_ . The key insight of BBS is that the bit-level sparsity can be explored in a symmetrical way, where less zero-bits implies more one-bits, and vice versa. This ensures that any bit vector can exhibit at least 50% BBS, which significantly improves the load balance of bit-serial computing while minimizing the number of ineffectual bit operations. Due to the balanced workload, BBS eliminates the expensive bit synchronization mechanism that is typically associated with prior bit-serial accelerators [1], [20], [26]. Furthermore, unlike previous bit-sparsity-aware works that only prune zero bit columns, BBS offers a new opportunity for model compression—it permits pruning a bit column with entirely zero-bits or entirely one-bits, which we call _bi-directional sparse bit columns_ . As shown in Fig. 1(c), by looking for an optimal way to generate 3 bi-directional sparse columns, we can achieve much lower MSE compared to merely pruning zero bit columns with the same compression ratio. Additionally, since BBS allows any bit significance to be one, it preserves all quantization levels of the original INT8 weight and yields much lower KL divergence w.r.t. the original numerical distribution pre-compression. Finally, the balanced nature of BBS can be exploited in a hardware-friendly manner to improve the performance and energy efficiency of bitserial accelerators. The main contributions of this work are summarized as follows: 

- 1) We introduce the new BBS concept, and demonstrate that BBS significantly improves the load balance of bitserial accelerators. 

- 2) We propose two bit-level _binary pruning_ strategies to enhance structured BBS. The binary pruning employs a new encoding scheme to reduce the memory footprint of a quantized DNN without the need of retraining. 

- 3) We design _BitVert_ , a bit-serial accelerator to exploit BBS for DNN acceleration. _BitVert_ adopts an efficient processing element (PE) with low hardware overhead for bit skipping, along with a channel-reordering mechanism to support binary pruning. 

552 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:15:39 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [483 x 219] intentionally omitted <==**

**----- Start of picture text -----**<br>
A Activation value (8b) ❷ Variable Shifter ❷ Multiplexer ❷ Ineffectual Ops<br>w [Weight zero bit] Time  Time<br>Time w Weight non-zero bit wwww01237777 wwww01236666 wwww0123 ❶ 5555 wwww Same weight 0123w4444 0wwww001233333 wwww01232222 wwww01231111 wwww8b01230000 ×  AAAA8b0123 w27 ❶ www02376 Different bit-significance 4 wwww01236453 wwww01235331 wwww0 123 2010 AAAA 0123 1b8b × × × ×  <<<<<<<< Time AAAA0123 ww02 ×  77 ww << 0266 7 ww0255 ... ww  8 lanes  1344 www123333 ... w02 << ww23 0 111bww × 13002b8b Different weights ❶ AAAA0123 ssss0123 ❶ wwww0123 Sign-magnitude format  4444 wwww01233333 wwww01232222 wwww01231111 wwww0 123 0000 AAAA 0123 1b8b × × × ×<br>(a) ❷ Ineffectual Ops (b) (c) (d)<br>Fig. 2: High-level computation flow of (a) bit-parallel PE, (b) Pragmatic [1], (c) Bitlet [26], (d) BitWave [39].<br>Through extensive evaluation on seven representative DNN 0.7 Value Bit (2's Comp) Bit (Sign Mag) BBS (2's Comp)<br>benchmarks, including both vision and language models, we 0.6<br>demonstrate that BitVert achieves up to 3 . 03 × speedup and 0.5<br>44 × energy saving compared to prior DNN accelerators, 0.4<br>having negligible accuracy loss ( <  0.5% on average) 0.3<br>0.2<br>with the preserved statistical characteristics of the 0.1<br>uncompressed model. 0<br>VGG-16 Resnet-34 Resnet-50 ViT-Small ViT-Base BERT<br>+<br>+<br>Sparsity<br>+ +<br>**----- End of picture text -----**<br>


Through extensive evaluation on seven representative DNN benchmarks, including both vision and language models, we demonstrate that _BitVert_ achieves up to 3 _._ 03 _×_ speedup and 2 _._ 44 _×_ energy saving compared to prior DNN accelerators, while having negligible accuracy loss ( _<_ 0.5% on average) together with the preserved statistical characteristics of the uncompressed model. 

## II. BACKGROUND AND RELATED WORKS 

Fig. 3: Comparison of inherent weight value sparsity, bit sparsity and BBS (with a bit-vector size of 8) in INT8 DNNs. 

## _A. Sparse Bit-serial Accelerators_ 

We first describe the computation flow of bit-parallel processing and recent sparse bit-serial accelerators [1], [26], [39] using a 4-way dot product example between 8-bit operands. We focus on weight sparsity in our discussion. In Fig. 2(a), a bit-parallel PE exploits bit-level parallelism by performing the multiplication between an 8-bit activation and all bits of the same weight, but leading to many ineffectual bit operations. Since zero bits do not contribute to the final result, it is desirable to skip as many zero bits as possible for improved performance and efficiency. 

ineffectual bit operations since only a bit column with all zero bits can be skipped during computation. On the top of these three design philosophies, our proposed _BitVert_ tries to balance the bit-serial workload while skipping as many sparse bits as possible. By extending bit sparsity to BBS, _BitVert_ skips zero bits when a bit column contains many zeros, while it switches to skip one bits when a bit column contains less zero bits. Section III details our BBS methodology. 

## _B. Rethinking Bit-level Sparsity_ 

Pragmatic [1] processes only non-zero bits of every weight as shown in Fig. 2(b). However, since different bit-significance can be processed simultaneously, Pragmatic requires a variable shifter after every bit-serial multiplier to synchronize the significance of essential bits. Bitlet [26] leverages the sparsity parallelism, motivated by the observation that every bit significance shows similar sparsity among a group of weights. As shown in Fig. 2(c), Bitlet digests multiple weights and activations, and computes every bit-significance independently. However, since every bit lane can absorb the essential bit from an arbitrary weight, Bitlet requires a large multiplexer ( _e.g._ , 64:1) to select the correct activation in every lane, leading to non-trivial hardware overhead (35 _._ 9% of the PE area as revealed by Bitlet’s breakdown report). 

While recent advances in PTQ can compress DNNs to 8-bit with little or no accuracy loss [5], [25], [32], [44], [45], the resulting weight tensor exhibits extremely low value sparsity. As shown in Fig. 3, the value-based weight sparsity is less than 5% in a series of popular 8-bit quantized DNNs. This is because that a well-designed PTQ algorithm tries to utilize all available quantization levels to minimize the quantization MSE compared to original floating-point models. On the other hand, the bit-level sparsity is inherently more abundant and can achieve around 50% in 2’s complement format. Owing to the facts that DNN weight tensors usually exhibit Gaussian-like distribution and most values tend to be small [16], [34], [46], the sign-magnitude binary representation yields even higher bit sparsity [2], [39] due to abundant zero bits at higher bit significance. However, adopting sign-magnitude arithmetic for bit-serial computing still has two challenges. First, every bitserial multiplier requires a 2’s complementer for partial sum generation, resulting in large area overhead [18]. Second, the irregular distribution of zero bits remains, leading to load imbalance and synchronization overhead. Whereas our proposed BBS maintains the 2’s complement binary representation, and treats zero or one that has a higher occurrence as sparse bits. Hence, BBS ensures that any bit-vector exhibits at least 50% 

Both Pragmatic and Bitlet suffer from load imbalance issues, where the latency of Pragmatic is dominated by the weight with the highest number of one bits, and the latency of Bitlet is dominated by the bit significance with the highest number of one bits. To address this, BitWave [39] attempts to skip zero bits at the coarse bit-column granularity, as illustrated in Fig. 2(d). Because most weight values are typically small in a DNN, BitWave relies on sign-magnitude format which inherently generates many zero bit columns. The bit column sparsity offers balanced workload, but inevitably leads to many 

553 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:15:39 UTC from IEEE Xplore.  Restrictions apply. 

|**❶ Remove Redundant Columns**<br>**Original Weight (2's Comp)**<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_= 5_**<br>**_= 4_**<br>**_= 7_**<br>**_= 5_**<br>_- _27 26 25 24<br>22<br>23<br>21 20<br>Signifcance|**❶ Remove Redundant Columns**<br>**Original Weight (2's Comp)**<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_= 5_**<br>**_= 4_**<br>**_= 7_**<br>**_= 5_**<br>_- _27 26 25 24<br>22<br>23<br>21 20<br>Signifcance|**❶ Remove Redundant Columns**<br>**Original Weight (2's Comp)**<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_= 5_**<br>**_= 4_**<br>**_= 7_**<br>**_= 5_**<br>_- _27 26 25 24<br>22<br>23<br>21 20<br>Signifcance|**❶ Remove Redundant Columns**<br>**Original Weight (2's Comp)**<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_= 5_**<br>**_= 4_**<br>**_= 7_**<br>**_= 5_**<br>_- _27 26 25 24<br>22<br>23<br>21 20<br>Signifcance|**❶ Remove Redundant Columns**<br>**Original Weight (2's Comp)**<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_= 5_**<br>**_= 4_**<br>**_= 7_**<br>**_= 5_**<br>_- _27 26 25 24<br>22<br>23<br>21 20<br>Signifcance|**❶ Remove Redundant Columns**<br>**Original Weight (2's Comp)**<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_= 5_**<br>**_= 4_**<br>**_= 7_**<br>**_= 5_**<br>_- _27 26 25 24<br>22<br>23<br>21 20<br>Signifcance|**❶ Remove Redundant Columns**<br>**Original Weight (2's Comp)**<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_= 5_**<br>**_= 4_**<br>**_= 7_**<br>**_= 5_**<br>_- _27 26 25 24<br>22<br>23<br>21 20<br>Signifcance|**❶ Remove Redundant Columns**<br>**Original Weight (2's Comp)**<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_= 5_**<br>**_= 4_**<br>**_= 7_**<br>**_= 5_**<br>_- _27 26 25 24<br>22<br>23<br>21 20<br>Signifcance|**❶ Remove Redundant Columns**<br>**Original Weight (2's Comp)**<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_= 5_**<br>**_= 4_**<br>**_= 7_**<br>**_= 5_**<br>_- _27 26 25 24<br>22<br>23<br>21 20<br>Signifcance|**❶ Remove Redundant Columns**<br>**Original Weight (2's Comp)**<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_-11_**<br>0<br>0<br>1<br>1<br>1<br>1<br>1<br>1<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_20_**<br>0<br>0<br>0<br>0<br>1<br>1<br>0<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_-57_**<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>1<br>0<br>0<br>1<br>0<br>0<br>**_= 5_**<br>**_= 4_**<br>**_= 7_**<br>**_= 5_**<br>_- _27 26 25 24<br>22<br>23<br>21 20<br>Signifcance|**❶ Remove Redundant Columns**|**❶ Remove Redundant Columns**|**❶ Remove Redundant Columns**|**❶ Remove Redundant Columns**|**❶ Remove Redundant Columns**|**❶ Remove Redundant Columns**|**❶ Remove Redundant Columns**|**❶ Remove Redundant Columns**|**❶ Remove Redundant Columns**|**❶ Remove Redundant Columns**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||0|0|0|0|1|1|0|1||**_13_**|0|0|0|0|1|1|0|1|**_= 5_**|
||1|1|0|0|0|1|1|1||**_-57_**|1|1|0|0|0|1|1|1|**_= 7_**|
||0|0|0|1|0|1|0|0||**_20_**|0|0|0|1|0|1|0|0|**_= 4_**|
||1|1|1|1|0|1|0|1||**_-11_**|1|1|1|1|0|1|0|1|**_= 5_**|
||||||||||||_- _27|26|25|24|23|22|21|20||



|**❷ Rounded Column Average**|**❷ Rounded Column Average**|**❷ Rounded Column Average**|**❷ Rounded Column Average**|**❷ Rounded Column Average**|**❷ Rounded Column Average**|**❷ Rounded Column Average**|**❷ Rounded Column Average**|**❷ Rounded Column Average**|**❸ Pruning and Compression**<br>+<br>0<br>1<br>**# Redundant Columns**<br>**Metadata**<br>**_-59_**<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>0<br>0|**❸ Pruning and Compression**<br>+<br>0<br>1<br>**# Redundant Columns**<br>**Metadata**<br>**_-59_**<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>0<br>0|**❸ Pruning and Compression**<br>+<br>0<br>1<br>**# Redundant Columns**<br>**Metadata**<br>**_-59_**<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>0<br>0|**❸ Pruning and Compression**<br>+<br>0<br>1<br>**# Redundant Columns**<br>**Metadata**<br>**_-59_**<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>0<br>0|**❸ Pruning and Compression**<br>+<br>0<br>1<br>**# Redundant Columns**<br>**Metadata**<br>**_-59_**<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>0<br>0|**❸ Pruning and Compression**<br>+<br>0<br>1<br>**# Redundant Columns**<br>**Metadata**<br>**_-59_**<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>0<br>0|**❸ Pruning and Compression**<br>+<br>0<br>1<br>**# Redundant Columns**<br>**Metadata**<br>**_-59_**<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>0<br>0|**❸ Pruning and Compression**<br>+<br>0<br>1<br>**# Redundant Columns**<br>**Metadata**<br>**_-59_**<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>0<br>0|**❸ Pruning and Compression**<br>+<br>0<br>1<br>**# Redundant Columns**<br>**Metadata**<br>**_-59_**<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>0<br>0|**❸ Pruning and Compression**<br>+<br>0<br>1<br>**# Redundant Columns**<br>**Metadata**<br>**_-59_**<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>0<br>0|**❸ Pruning and Compression**<br>+<br>0<br>1<br>**# Redundant Columns**<br>**Metadata**<br>**_-59_**<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>0<br>0|**❸ Pruning and Compression**<br>+<br>0<br>1<br>**# Redundant Columns**<br>**Metadata**<br>**_-59_**<br>0<br>1<br>0<br>0<br>**_13_**<br>1<br>0<br>0<br>0|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**_13_**|0|0|0|1|1|0|1|**_= 5_**|**_13_**|0|0|0|1||||||||
|**_-59_**|1|0|0|0|1|0|1|**_= 5_**|**_-59_**|1|0|0|0||0<br>1<br>**Metadata**||||||
|**_21_**|0|0|1|0|1|0|1|**_= 5_**|**_21_**|0|0|1|0||0|0|0|1|0|1|
|**_-11_**|1|1|1|0|1|0|1|**_= 5_**|**_-11_**|1|1|1|0||||||||
||_- _26|25|24|23|22|21|20|||_- _26|25|24|23||||||||



Fig. 4: Example of bit-level binary pruning with rounded column averaging to generate 4 sparse bit columns. 

bit sparsity, resulting in higher total bit sparsity than signmagnitude format while achieving balanced workload across different PEs. 

## III. BBS: BI-DIRECTIONAL BIT-LEVEL SPARSITY 

In this section, we first introduce the concept of BBS based on 2’s complement binary representation. Next, we present _binary pruning_ , a technique that modifies the original weight tensor to generate more structured BBS, together with a new encoding scheme that provides an extra opportunity for model compression. Finally, we propose a hardware-aware strategy to compress different weight channels of a DNN model based on the global awareness of pruning sensitivity, which can achieve favorable accuracy-compression trade-offs. 

original one-bits become sparse, and subtract the bit-serial dot product from[�] _[N] i_ =0 _[−]_[1] _[A][i]_[.][Since][both][zero][and][one][can][become] sparse bits, we call this _bi-directional bit sparsity (BBS)_ . 

The idea of BBS can effectively improve the load balance of bit-serial computing. Although there is _∼_ 50% zero bit sparsity in 2’s complement format and more than 50% zero bit sparsity in sign-magnitude format (Fig. 3), the sparsity within a bit-vector is unpredictable. Moreover, because bitserial computing relies on strongly increased parallelism to simultaneously process many bit-vectors from different weight groups, any bit-vector with low zero bit sparsity will hamper the performance of the whole PE array. On the other hand, BBS ensures at least 50% sparsity in a bit-vector of arbitrary length, achieving balanced workload during parallel execution while skipping as many ineffectual bit operations as possible. 

## _A. BBS Theorem_ 

Without loss of generality, we describe BBS using a dot product operation that multiplies a group of _N_ weights ( _W_ ) and activations ( _A_ ) in _p_ -bit precision, where _N_ is referred to as the _group size_ . In the rest of this paper, we use the term “ _group_ ” to refer to multiple weights or activations that contribute to the same dot product output. The dot product operation can be formally written as: 

**==> picture [197 x 29] intentionally omitted <==**

where _Wi[b]_[is][the] _[b][th]_[bit][of] _[W][i]_[.][Since][any][bit][of] _[W]_[can][only] be one or zero, the second partial sum on the right-hand side of Eq. 1 can be re-organized as: 

**==> picture [190 x 67] intentionally omitted <==**

From Eq. 2 and 3, we can infer that instead of adding the effectual activations associated with non-zero weight bits, the same result can be obtained by subtracting the activations indicated by zero weight bits from the sum of all activations, which is a constant for a given group. Since more zero-bits in a vector implies less one-bits, Eq. 2 and Eq. 3 always process no more than half of the bits—when there are more than 50% zero-bits in a bit-vector, the computation can skip them as in conventional bit-serial accelerators. But if there is less than 50% bit sparsity, the bit-vector can be _inverted_ so that the 

## _B. Bit-level Binary Pruning_ 

In addition to balanced bit sparsity, BBS offers a new opportunity for model compression through _binary pruning_ — which can prune a bit column that contains all zero-bits or all one-bits within a weight group. Specifically, Eq. 2 implies that if all weight bits at a bit significance are zero, then the bitserial dot product at that significance is simply zero. Similarly, Eq. 3 implies that if all weight bits at a significance are one, then the bit-serial dot product at that significance is the sum of activations in the group. As a result, a bi-directional sparse bit column can be compressed to just one bit that indicates whether its bit-serial dot product produces zero or sum of activations. Based on this observation, we propose two BBSenhancing strategies to generate more bi-directional sparse bit columns in the original weight group, which can be effectively pruned through a new encoding scheme. 

**BBS with Rounded Averaging** Fig. 4 describes the procedure of the first BBS-enhancing strategy, _rounded averaging_ , using a group of 4 weights. Given the target number of sparse bit columns (4 in this example), Step 1 identifies if there are _redundant_ bit columns that immediately follow the mostsignificant column with the same content (e.g., the second bit column). Removing the redundant columns does not affect the original weight values as long as the remaining bits are interpreted as 2’s complement format. For instance, the decimal number _−_ 57 in 8-bit 2’s complement format is 11000111 _b_ , where the most-significant bit is multiplied by _−_ 2[7] . Removing the second bit leads to a 7-bit number 1000111 _b_ , which is still equal to _−_ 57 if the most-significant bit is multiplied by _−_ 2[6] . After pruning the redundant column, the required number 

554 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:15:39 UTC from IEEE Xplore.  Restrictions apply. 

|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**Original Weight (2's Comp)**<br>**_-7_**<br>1<br>0<br>1<br>1<br>1<br>0<br>1<br>1<br>**_1_**<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>**_-20_**<br>1<br>0<br>0<br>1<br>0<br>1<br>1<br>1<br>**_81_**<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>0<br>**❶ Add****_-14_ to Weight**<br>**_-21_**<br>1<br>1<br>1<br>1<br>0<br>0<br>1<br>1<br>**_-13_**<br>0<br>1<br>1<br>1<br>1<br>0<br>1<br>1<br>**_-34_**<br>1<br>1<br>0<br>1<br>1<br>1<br>1<br>0<br>**_67_**<br>0<br>1<br>1<br>0<br>0<br>0<br>1<br>0<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**<br>**_−14  =_**|**❷ Generate Zero Columns**|**❷ Generate Zero Columns**|**❷ Generate Zero Columns**|**❷ Generate Zero Columns**|**❷ Generate Zero Columns**|**❷ Generate Zero Columns**|**❷ Generate Zero Columns**|**❷ Generate Zero Columns**|**❷ Generate Zero Columns**|**❷ Generate Zero Columns**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|**❸ Actual Weight with BBS Compression**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||0|1|0|1|0|0|0|1|**_67_**<br>**_−14  =_**|0|1|0|0|0|0|1|1||**_64_**|0|1|0|0|0|0|0|0||**_78_**|0|1|0|0|1|1|1|0|0<br>0<br>**# Redundant Columns**<br>**Metadata**|||||||
||1|1|1|0|1|1|0|0|**_-34_**<br>**_−14  =_**|1|1|0|1|1|1|1|0||**_-32_**|1|1|1|0|0|0|0|0||**_-18_**|1|1|1|0|1|1|1|0||0|0|**Metadata**||||
||0|0|0|0|0|0|0|1|**_-13_**<br>**_−14  =_**|1|1|1|1|0|0|1|1||**_-16_**|1|1|1|1|0|0|0|0||**_-2_**|1|1|1|1|1|1|1|0||0|0|1|1|1|0|
||1|1|1|1|1|0|0|1|**_-21_**<br>**_−14  =_**|1|1|1|0|1|0|1|1||**_-16_**|1|1|1|1|0|0|0|0||**_-2_**|1|1|1|1|1|1|1|0|**BBS Constant = 14**|||||||
||||||||||||||||||||||||||||||||||||||||||||||



Fig. 5: An example of bit-level binary pruning with zero-point shifting to generate 4 sparse bit columns. 

of bi-directional sparse columns to be generated is 3. These sparse columns are always generated from the lower significant bits, since modifying higher bit significance will increase the MSE exponentially. In Step 2 , this is achieved by calculating the rounded average of the values represented by the 3 lower significant bits of original weights. Essentially, this is replacing the 3 lower significant bits of all weights with a 3-bit constant while minimizing the MSE. Finally, Step 3 compresses the original weight group by storing only the remaining 4 bit columns and an 8-bit encoding metadata. 

**BBS Compression Encoding** The encoding metadata contains 2 bits to specify the number of redundant columns, which can vary from 0 to 3, and 6 bits to store the BBS constant. The size of the metadata is chosen empirically. First, although there may be more than 3 redundant columns in a group, we find that this probability is extremely low for a large group size ( _e.g._ , 32) which amortizes the cost of metadata. If there are more than 3 redundant columns, we simply prune the first 3 and average additional lower significant columns instead. Second, using more than 6 bits to store the constant is also unnecessary since pruning 7 columns of an 8-bit tensor leaves only one effective bit, while pruning 8 columns means replacing all weights with the same 8-bit constant. Both situations can lead to unacceptable accuracy loss. 

**BBS with Zero-point Shifting** The rounded column averaging strategy is particularly suitable for pruning a small number of bit columns, where the lower significant bits within a group are likely to have similar values. However, for more eager compression, _i.e._ , pruning many columns, simply taking the rounded average over many lower significant bits of a group may lead to large MSE. Here is a simple example: assume we want to average only the least significant bit within a group of weights, then some weights will have no error after rounded averaging. On the other hand, if we average 4 lower significant bits, then all weights may produce error since any weight can have a different value in the 4 lower significant bits. 

To address this, we propose a second BBS-enhancing strategy called _zero-point shifting_ . The idea is to add an optimal constant to the original weight group ( _i.e._ , shifting its zeropoint), which in turn facilitates the generation of sparse bit columns in the new weight group while minimizing the MSE. Fig. 5 exemplifies this procedure for generating 4 sparse bit columns. In Step 1 , assume a constant _−_ 14 is added to the original weight, which changes the binary content of all numbers. Fortunately, the change of binary content makes it easier to generate zero columns in lower significant bits. As 

|**Algorithm 1:** Finding the optimal constant for zero-point shifting.|**Algorithm 1:** Finding the optimal constant for zero-point shifting.|**Algorithm 1:** Finding the optimal constant for zero-point shifting.|**Algorithm 1:** Finding the optimal constant for zero-point shifting.|**Algorithm 1:** Finding the optimal constant for zero-point shifting.|
|---|---|---|---|---|
||**Input**<br>**:**|||Weight group: _W_, BBS constant precision: _p_,|
|||||target number of sparse bit columns: _N_|
||**Output :**|||Compressed weight: _WC_, metadata: _D_|
|**1 **|**def** Compress(_W_, _N_, _p_)**:**||||
|**2**||bestMSE = _∞_;|||
|**3**||**for** constant = _−_2_p−_1 **to** 2_p−_1 _−_1 **do**|||
|**4**|||_Wtmp_ = Clip(_W_ + constant)||
||||// Get number of redundant columns||
|**5**|||numRedunCol = GetNumRedunCol(_Wtmp_)||
|**6**|||_Wtmp_ = RemoveRedunCol(_Wtmp_, numRedunCol)||
||||// Generate zero sparse columns||
|**7**|||numSparseCol = _N −_numRedunCol||
|**8**|||_Wtmp_ = GenSparseCol(_Wtmp_, numSparseCol)||
|**9**|||newMSE = _|Wtmp −W|_2||
|**10**|||**if** newMSE _<_ bestMSE **then**||
|**11**||||bestMSE = newMSE|
|**12**||||_WC_ = _Wtmp_|
|**13**||||_D_ = _{_numRedunCol , constant_}_|
||||||
|**14**||**return** _WC, D_|||



shown in Step 2 , to minimize the MSE when pruning the 4 lower significant bit columns, a number can either directly zero out the 4 lower bits (e.g., the first number changes from 67 to 64), or round up to the higher bit significance (e.g., the second number changes from _−_ 34 to _−_ 32). Finally, Step 3 shows the actual values after binary pruning and stores the new zero-point in the encoding metadata. 

Algo. 1 details the algorithm to find the optimal BBS constant for a weight group. Given the precision of the constant (6-bit in our proposed BBS encoding), the algorithm iterates through all possible constants (Line 3). In every iteration, it adds the current constant to the original weight group, followed by clipping to avoid overflow (Line 4). Next, similar to _rounded averaging_ , we calculate the number of redundant columns, and generate required number of sparse columns while minimizing MSE (Line 5 – 7). Since the best constant will be stored in the BBS constant region of the metadata, we only generate zero sparse bit columns (Line 8) so that no extra encoding information is needed. Lastly, the algorithm checks whether the current constant results in lower MSE and updates the weight group and metadata accordingly (Line 9 – 13). 

Although Algo. 1 describes the procedure using a single weight group, the whole algorithm can be vectorized to find the optimal constant of all groups within a DNN layer simultaneously. During real implementation, the algorithm takes several milliseconds to several seconds per layer (totally _∼_ 15s to 

555 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:15:39 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [237 x 177] intentionally omitted <==**

Fig. 6: Normalized KL divergence (lower is better) of different bit-level pruning techniques with a weight group size of 32. 

compress the whole ResNet50) on a single Nvidia RTX 3090 GPU. Hence, the proposed bit-level binary pruning method exhibits high efficiency and fast compression compared to prior quantization-oriented algorithms [5], [45], [46]. 

**Rationality of Binary Pruning** To demonstrate the rationality of the proposed two binary pruning strategies compared to previous zero-bit-only pruning [23], [35], [39], we apply the three techniques to compress the quantized 8-bit ResNet-34 and ViT-Base. Fig. 6 shows the resulting KL divergence of different methods after pruning 2 and 4 bit columns with a weight group size of 32. The KL divergence is a common metric to quantify the difference between two distributions [11], [17]. A lower KL divergence indicates that the compressed weight tensor can better preserve the information of the original 8-bit weight, thus achieving better inference accuracy (evaluated in Section V-B). 

Specifically, Fig. 6 shows that when pruning 2 bit columns, _rounded averaging_ consistently outperforms other approaches. The reason is that different weights within a group are likely to have similar values in the lower significant bits. On the other hand, _zero-point shifting_ yields much lower KL divergence when pruning 4 bit columns. This is because it can better exploit the binary characteristics of a weight group to find the optimal zero point that facilitates the generation of more sparse bit columns. Furthermore, the proposed binary pruning permits the existence of both zero and one in any bit significance after compression, thus are able to preserve all quantization levels of the original 8-bit weights as opposed to zero-bit-only pruning. As a result, both of our strategies show significant improvements when applied to a large number of bit columns. 

## _C. Hardware-aware Global Binary Pruning_ 

So far, we have described binary pruning at the group level. In order to fully exploit the structured BBS sparsity induced by binary pruning while mitigating the accuracy loss for the whole DNN, we propose a hardware-aware global binary pruning approach at the _per-channel_ granularity. Specifically, 

**==> picture [238 x 237] intentionally omitted <==**

**----- Start of picture text -----**<br>
Algorithm 2: Global binary pruning.<br>Input : Model: M , per-channel scaling factors: S<br>threshold: β , hardware parameter: CH<br>Output : Pruned model: MP<br>1 def GlobalPrune( M , S , β , CH )  :<br>// Global channel sorting<br>2 channelSorted = SortChannel( M.channel, S  )<br>3 sensChannel = channelSorted [ 1 : β × Length ( S ) ]<br>4 for L in M.layers do<br>// Ensure every layer has a multiple<br>of CH sensitive channels<br>5 layerChannel = SortChannel( L.channel, S [ L ] )<br>6 numSens = Count( layerChannel ∩ sensChannel )<br>7 numSens = Ceiling( numSens / CH  )  × CH<br>// Get sensitive channels of layer L<br>8 topChannel = layerChannel [ 1 : numSens ]<br>9 sensChannel = sensChannel ∪ topChannel<br>10 normalChannel = M.channel − sensChannel<br>11 if eagerCompression then<br>12 MP = RoundedAveraging(normalChannel)<br>13 else<br>14 MP = ZeroPointShifting(normalChannel)<br>15 return MP<br>**----- End of picture text -----**<br>


we find that the pruning sensitivity of different weight channels can be effectively quantified through magnitude-based proxies. For example, in convolutional neural networks, the sensitive filters ( _i.e.,_ weight channels) usually contain many outliers with large magnitude. More specifically, in per-channel quantized DNNs, the sensitive channels of a weight tensor will have large scaling factors to accommodate these outliers [27], [44]. The per-channel weight quantization has been widely adopted to achieve high accuracy in state-of-the-art DNN accelerators [3], [16] and acceleration frameworks such as TensorRT [33]. Therefore, we consider per-channel quantized 8-bit DNNs as the baseline for global binary pruning[1] . 

To apply global binary pruning, we define a hyperparameter _β_ to specify the minimum percentage of sensitive weight channels. Also, we define a hardware-aware parameter _CH_ , which specifies the number of weight channels processed in parallel during hardware acceleration ( _e.g._ , _CH_ = 32 in our _BitVert_ accelerator). Algo. 2 details the procedure of global binary pruning. The algorithm starts with global channel sorting to identify _β_ sensitive channels based on the scaling factors (Line 1 – 2). For every layer, we force the number of sensitive channels to be a multiple of _CH_ (Line 4 – 9). For example, in the convolution layer, if the number of sensitive filters is less than _CH_ after global channel sorting, then we simply select _CH_ filters with the highest scaling factors as new sensitive channels. Finally, we apply binary pruning to the remaining channels (Line 10 – 14), which can either prune a different number of bit columns for different layers [39] or prune the same number of bit columns for all layers. 

> 1For 8-bit DNNs that do not use per-channel quantization, other channel importance proxies such as the standard deviation of a weight channel can also be used to identify sensitive channels. 

556 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:15:39 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [484 x 227] intentionally omitted <==**

**----- Start of picture text -----**<br>
15 ∑ 0AAAA1501i 8 12 ❶ Act Select 8 1 ❷ Bit-serial Multiplier - 12 ❹ BBS Multiplier bbs_const ×  6 prod 18 + ❺ Accumulation out A ∑ 070...AAseli 04 ❶ 11 Term Select 3 8 ×4 1 ×4 ❷ Bit-serial Multiplier - 11 ❹ + BBS Multiplier bbs_const ×  3 << prod18<br>selsel01 4×88 ×8 psum_selpsum neg col_idx << 3 19 24 A3...Asel77 3 8×4 1 ×4 psum_selpsum_sel1101 + psum12<br>1 is_msb out_prev A11...A15 8 -<br>sel7 4 ❸ Single Shift ∑ 15 Ai 11 1<br>8 8 8<br>val0 ... val7 (a) val0 ... val7 (b)<br>Fig. 7: BitVert PE: (a) baseline design, (b) modified design.<br>The identification of sensitive channels further reduces the PopCount > 4 sel0 val0 sel1 val1 sel2 val2 sel3 val3 = 0 # RedunCol<br>and KL divergence while eliminating the need for w0 0 1 psum_sel 1 3 1 3 1 3 1 3 1 0111b -<br>resource-intensive= 10%theDNNremaining orbenchmarks 20% while pruning a large number of bit columnschannels.and= 10%theDNNremaining orbenchmarks 20% while pruning a large number of bit columnschannels.andtheDNNremaining orbenchmarks 20% while pruning a large number of bit columnschannels.andDNNremaining orbenchmarks 20% while pruning a large number of bit columnschannels.andremaining orbenchmarks 20% while pruning a large number of bit columnschannels.and orbenchmarks 20% while pruning a large number of bit columnschannels.andbenchmarks 20% while pruning a large number of bit columnschannels.and 20% while pruning a large number of bit columnschannels.and while pruning a large number of bit columnschannels.andchannels.andtime-consuming(SectionHowever,V-A),sinceretraining.we(SectionHowever,V-A),sinceretraining.weHowever,V-A),sinceretraining.weV-A),sinceretraining.we),sinceretraining.wesinceretraining.wetheare time-consuming(SectionHowever,V-A),sinceretraining.we(SectionHowever,V-A),sinceretraining.weHowever,V-A),sinceretraining.weV-A),sinceretraining.we),sinceretraining.wesinceretraining.wetheare theare locationsableIn mosttoto setofofofofof wwwwww123456 101101 010010 0100 01001 00010 0000 ... -mux1<br>channels are random within a layer, two challenges w7 Bit Column Selection 1 0 Activation Index Generation 0 Shift Ctrl col_idx<br>1 - 1 5-<br>... 16 ... ...<br>... 1 5-<br>...<br>... 1 16- ... 1 5- ... ...<br>1 2-<br>Priority Encoder Priority Encoder Priority Encoder Priority Encoder<br>+ +<br>+<br>+<br>**----- End of picture text -----**<br>


The identification of sensitive channels further reduces the MSE and KL divergence while eliminating the need for our and retraining.wewe are ableInIn mosttoto _β_ inresource-intensive= 10%theDNNremaining orbenchmarks 20% while pruning a large number of bit columnschannels.and= 10%theDNNremaining orbenchmarks 20% while pruning a large number of bit columnschannels.andtheDNNremaining orbenchmarks 20% while pruning a large number of bit columnschannels.andDNNremaining orbenchmarks 20% while pruning a large number of bit columnschannels.andremaining orbenchmarks 20% while pruning a large number of bit columnschannels.and orbenchmarks 20% while pruning a large number of bit columnschannels.andbenchmarks 20% while pruning a large number of bit columnschannels.and 20% while pruning a large number of bit columnschannels.and while pruning a large number of bit columnschannels.andchannels.andtime-consuming(SectionHowever,V-A),sinceretraining.we(SectionHowever,V-A),sinceretraining.weHowever,V-A),sinceretraining.weV-A),sinceretraining.we),sinceretraining.wesinceretraining.wetheare locationsableIn setofofofofof sensitive channels are random within a layer, two challenges arise for efficient hardware acceleration. First, identifying the location of sensitive channels requires significant indexing overhead. Second, different precision will cause unaligned memory access to the weight tensor in DRAM. The proposed _BitVert_ accelerator addresses these challenges through a channel-reordering mechanism as will be discussed shortly. 

Fig. 8: BitVert scheduler. 

until all bit-columns belonging to the same weight group are processed. The control signals such as _sel_ , _val_ , and _col idx_ are updated by the _BitVert_ scheduler in every cycle (described in Section IV-B). 

## IV. BITVERT HARDWARE ARCHITECTURE 

Due to the random distribution of effectual bits within a weight bit-column, the baseline PE accounts for the worst case by using a 16:1 mux for every activation term. Since BBS guarantees at least 50% sparsity in a bit-vector of arbitrary length, it is possible to reduce the mux cost with a smaller group size. Based on this observation, we propose a modified PE that computes bit-serial multiplication within a smaller _subgroup_ as shown in Fig. 7(b). The sub-group size is a design parameter that offers a trade-off between area and power. A smaller sub-group can reduce the mux cost but requires more subtractors. Therefore, we conduct a PE design space exploration (Section V-E) and choose a sub-group size of 8 in our design. Furthermore, because the PE supports 50% bit sparsity, at most 4 activations will be selected within a subgroup. In the worst case, the selected activations within the sub-group _{A_ 0 _, ..., A_ 7 _}_ will be _{A_ 4 _, A_ 5 _, A_ 6 _, A_ 7 _}_ . Hence, we only need four 5:1 muxes to locate all effectual activations, where the first mux selects among _{A_ 0 _, ..., A_ 4 _}_ , the second mux selects among _{A_ 1 _, ..., A_ 5 _}_ , and so on. Using 5:1 muxes further reduces the PE area compared to 8:1 muxes. 

To fully exploit the potential of BBS and binary pruning, we design a bit-serial accelerator, named _BitVert_ , which includes an efficient PE and scheduler to support BBS with compression, along with the channel reordering mechanism for hardware-aware global binary pruning. 

## _A. BitVert Processing Element_ 

The _BitVert_ PE performs bit-serial multiplication between a group of 16 weights and activations, where weights are processed bit-serially. Fig. 7(a) shows a baseline _BitVert_ PE that performs the computation in 5 steps. Step 1 receives 16 activations _A_ 0 _, ..., A_ 15 and selects 8 of them based on _sel_ 0 _, ..., sel_ 7 that indicates the position of effectual bits in the weight bit-vector. Step 2 performs bit-serial multiplication using valid signals _val_ 0 _, ..., val_ 7 in case there are less than 8 effectual bits ( _i.e._ , more than 50% sparsity in the weight bitcolumn). A subtractor subtracts the adder tree result from the sum of activations (Eq. 2), followed by a mux to select the partial sum. Step 3 then shifts the partial sum based on the column index _col idx_ that specifies the significance of current weight bits. The _col idx_ can vary across different groups according to the number of redundant columns during binary pruning (Section III-B). Recall that BBS compression stores a constant, whose “0” bit indicates a bit-column of all zerobits and “1” bit indicates a bit-column of all one-bits. Hence, Step 4 multiples this constant with the sum of activations. Finally, the product and bit-serial partial sum are accumulated in Step 5 . The activations are reused for multiple clock cycles 

It is also possible to reduce the cost of the BBS multiplier in Step 4 . Since BBS can prune a maximum of 6 bit columns in a weight group (Section III-B), it requires at least 2 cycles to process the remaining columns when the weight precision is 8 bits. This allows time-multiplexing the BBS multiplier by multiplying 3 bits per cycle, followed by a shifter to align the significance. Section V-E evaluates the reduction in PE area overhead achieved by the proposed optimization. 

557 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:15:39 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [237 x 112] intentionally omitted <==**

**----- Start of picture text -----**<br>
8-bit Channel Weight  Output 1<br>Order 1 × =<br>Chunk 1<br>K Input +<br>Chunk 2<br>Weight  Correct Wrong<br>Order 2 × =<br>6-bit Channel (a) (b) Output 2<br>ChannelOriginal  Reordered Channel Input Output Channel Original  Restored Output<br>Index<br>× = Buffer<br>(c)<br>**----- End of picture text -----**<br>


Fig. 9: Channel reordering: (a) Store channels with the same precision in the same memory chunk. (b) Two weight tensors in a residual block with different channel orders can lead to the wrong result when processing the same input. (c) Unshuffle the output to restore the original channel order. 

## _B. BitVert Scheduler_ 

_BitVert_ adopts a low-cost scheduler to control the operation within a PE, as illustrated in Fig. 8. To control the bit-serial dot product, the scheduler first identifies whether there are more zero bits in a bit column. It then sends the original or inverted bit column to a series of 4 priority encoders. Every priority encoder receives 5 consecutive bits from the weight bit column. For example, the first priority encoder receives _{w_ 0 _, .., w_ 4 _}_ , the second receives _{w_ 1 _, .., w_ 5 _}_ , and so on. The encoder detects the location of the first “1” bit in the received 5-bit vector. If exists, it will mask the detected “1” bit and sends the remaining bits to the next encoder. On the other hand, if the received 5-bit vector contains all zero-bits, the encoder will signal _val_ = 0 to disable the corresponding bitserial multiplier in the PE. 

The scheduler also generates the _col idx_ signal to control the shifting of bit-serial multiplier in every PE. When a new dot product begins, the scheduler receives the BBS metadata which contains the number of redundant columns, # _RedunCol_ , in a weight group. The highest bit significance of the compressed weight group indicates the initial _col idx_ and is obtained by subtracting the number of redundant columns from 7 (i.e., the highest bit significance of uncompressed weight). The _col idx_ is updated in every cycle by subtracting one until the bit-serial bot product completes. 

## _C. Channel Reordering_ 

With per-channel global binary pruning, the sensitive and normal channels will have different precision, resulting in unaligned memory layout. To address this issue, we adopt a channel reordering mechanism as shown in Fig. 9(a). There are 6 weight channels in this example, and channels with the same precision are grouped together and stored in a memory chunk to avoid unaligned access. Recall from Section III-C that the proposed global binary pruning is hardware-aware, which forces the number of sensitive channels in every layer to be a multiple of the number of channels processed in parallel. Therefore, the grouped channels can be efficiently accessed by _BitVert_ to ensure full hardware utilization. 

The channel reordering mechanism has also been explored in SparTen’s greedy balancing [13]. However, the reordering 

**==> picture [236 x 112] intentionally omitted <==**

**----- Start of picture text -----**<br>
Weight Buffer<br>Metadata<br>BitVert Scheduler<br>Buffer sel val bbs_const Idx BufferChannel<br>PE  PE  PE<br>Input  (0, 0) (0, 1) (0, 31)<br>Buffer<br>PE  PE  PE<br>(1, 0) (1, 1) (1, 31) Output<br>Buffer<br>16 × 32<br>Group ∑A  PE  PE  PE<br>Generator (15, 0) (15, 1) (15, 31)<br>**----- End of picture text -----**<br>


Fig. 10: BitVert accelerator. 

criteria is completely different. SparTen is a value-based sparse DNN accelerator that reorders weight channels based on their sparsity, while _BitVert_ groups channels based on their sensitivity to binary pruning. Furthermore, SparTen statically unshuffles the next layer’s weights in software, which may not guarantee the correctness when different weight tensors need to process the same input. Consider the example shown in Fig. 9(b), where two weight tensors multiply the same input and generate two output tensors that require element-wise addition ( _e.g._ , as in the residual block of ResNet). SparTen statically unshuffles the two weight tensors along the _K_ - dimension to align with the channel order of the previous layer, but the different channel order between the two weight tensors remains, which produce two output tensors with different orders. In this example, the second element of output 2 is supposed to be added with the third element of output 1, while a conventional design like SparTen will add the same position of two output tensors, leading to the wrong result. 

To solve the above issue, we propose to unshuffle the output tensor when writing back to memory. As shown in Fig. 9(c), after completing the whole dot product between the input tensor and reordered weight, the outputs are directly restored to the original channel order. This restoring only needs to know the original index of every weight channel to calculate the corresponding memory address for storing the final outputs. Fortunately, since a weight channel usually contains hundreds to thousands of values, the overhead of storing one index per channel is trivial. Moreover, because the same weight channel can process many inputs (3 in this example) to compute many outputs simultaneously, these outputs can be unshuffled together to amortize the cost of channel reordering. 

## _D. BitVert Accelerator_ 

Fig. 10 shows the overall architecture of the _BitVert_ accelerator. The 16 _×_ 32 PE array adopts an output-stationary dataflow, and exploits both weight-sharing and input-sharing by processing 32 weight channels and 16 input windows in parallel. The weight and input buffers are banked to provide adequate bandwidth for the access from PEs. Outputs are read out of the PE array and written to the output buffer, one column at a time. Additionally, _BitVert_ incorporates a metadata buffer to store BBS compression metadata, and a channel index buffer to store the original index of weight channels being processed. The Σ _A_ generator calculates the sum of input activations for 

558 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:15:39 UTC from IEEE Xplore.  Restrictions apply. 

BBS-based bit-serial multiplication inside the PE. Since the same input group is multiplied by 32 weight channels, the Σ _A_ generator incurs practically no overhead. 

## V. EVALUATION 

## _A. Experimental Methodology_ 

**DNN Benchmarks** We evaluate seven representative DNN models, including CNNs and transformer networks as summarized in Table I. For CNNs, we evaluate VGG-16, ResNet-34 and ResNet-50 on the ImageNet-1K dataset. For transformers, we choose two vision transformers, ViT-Small and ViT-Base, as well as BERT on MRPC and SST2 tasks from the GLUE dataset [41]. We obtain pre-trained CNNs and transformers from PyTorch Library and HuggingFace, respectively. We then conduct post-training per-channel quantization to obtain the baseline 8-bit models, which shows negligible accuracy loss compared to FP32 models. The 8-bit models are used to evaluate the proposed binary pruning technique and _BitVert_ accelerator. For every model, we apply two levels of binary pruning, _conservative_ (cons) and _moderate_ (mod), with a weight group size of 32. For conservative pruning, 10% sensitive channels are maintained at 8 bits and the remaining channels have 2 bit-columns pruned using the rounded averaging strategy. For moderate pruning, 20% sensitive channels are maintained at 8 bits and the remaining channels have 4 bit-columns pruned using the zero-point shifting strategy. 

**Accelerator Baselines** We compare _BitVert_ against six DNN accelerators, including four bit-serial accelerators: Stripes [19], Pragmatic [1], Bitlet [26], BitWave [39], and two value-based accelerators: SparTen [13], ANT [16]. Stripes is an early bitserial accelerator that exploits reduced precision for DNN computation, yet it mainly relies on 16-bit models and does not consider below-8-bit compression. Therefore, we treat Stripes as a dense bit-serial accelerator and use our baseline 8-bit models to evaluate its performance. Pragmatic and Bitlet target zero-bit skipping during on-chip computation only, while BitWave enhances structured bit-column sparsity to save both computation and memory access. SparTen exploits two-sided value sparsity for DNN acceleration. ANT combines different datatypes in a unified manner for low-bit DNN acceleration. We use 6-bit precision to evaluate ANT, a configuration demonstrated by ANT to maintain acceptable accuracy without the need of retraining. 

**Implementation** We implement the proposed binary pruning algotirhm in Pytorch. We design the _BitVert_ accelerator at RTL-level using SystemVerilog and synthesize it with Synopsys Design Compiler in TSMC 28nm technology to find 

|Type|CNN|CNN|Transformer|Transformer|Transformer|
|---|---|---|---|---|---|
|Model|VGG-16|ResNet-34 / 50|ViT-S / B|BERT||
|Dataset|ImageNet|||MRPC|SST2|
|FP32 Acc %|73.36|73.31 / 76.13|80.16 / 84.54|90.7|91.8|
|INT8 Acc %|73.35|73.39 / 76.17|80.05 / 84.52|90.4|91.63|



TABLE I. Summary of evaluated models and datasets. 

**==> picture [212 x 90] intentionally omitted <==**

Fig. 11: Comparison of accuracy loss between PTQ, BitWave and BBS under conservative (cons) and moderate (mod) compression. 

area. We use Synopsys VCS to generate data-driven activity factors at 800 MHz for power estimation. The area and power of on-chip SRAM buffer are modelled with CACTI [4]. To estimate the DRAM power, we use the DDR3 model from DRAMSim3 [22]. For the end-to-end performance evaluation of _BitVert_ and other baseline accelerators, we develop cycleaccurate simulators to model the execution time. To ensure a fair comparison, all accelerators are scaled to contain the same number of multipliers, where an 8-bit multiplier is equivalent to eight bit-serial multipliers. For on-chip SRAM, we equip ANT and all bit-serial accelerators with 256 KB activation buffer and 256 KB weight buffer. For SparTen, we reduce the size of its on-chip buffer due to the existence of the local buffer inside every PE. 

## _B. Accuracy Comparison_ 

We first evaluate the accuracy impact of BBS binary pruning compared to naive PTQ and BitWave’s bit-flip strategy [39] for compression below 8-bit. When using PTQ for compression, we follow the widely-used calibration [10] by calibrating the quantization parameters based on a subset (1024 images) of the ImageNet dataset. In particular, conventional PTQ relies on the calibration dataset to ensure the optimized quantization parameters and accuracy, while the naive data-free quantization leads to significant accuracy degradation ( _>_ 10%). On the contrary, the proposed BBS compresses the model to lower precision **without** any calibration dataset. For both PTQ and BitWave, we use the same setting as BBS by maintaining 20% and 10% sensitive channels for moderate and conservative pruning, respectively. This ensures that our accuracy benefits purely come from the proposed binary pruning. 

Fig. 11 shows the accuracy impact of applying different approaches on the baseline DNNs. On average, the conservative and moderate binary pruning can compress the memory footprint of the baseline 8-bit DNNs by 1 _._ 29 _×_ and 1 _._ 66 _×_ , while incurring an accuracy loss of only 0 _._ 25% and 0 _._ 45%, respectively. Both BitWave and BBS with moderate pruning can attain higher accuracy than PTQ. These accuracy improvements stem from their ability to exploit fine-grained bitlevel redundancy, thereby preserving more information from the original 8-bit models. Additionally, the proposed binary pruning consistently outperforms BitWave. This is because BBS allows any bit significance to be zero or one, thus retaining all quantization levels of the 8-bit precision. 

559 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:15:39 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [485 x 260] intentionally omitted <==**

**----- Start of picture text -----**<br>
SparTen ANT Stripes Pragmatic Bitlet BitWave BitVert (cons) BitVert (mod)<br>3.5<br>3.0<br>2.5<br>2.0<br>1.5<br>1.0<br>0.5<br>0.0<br>VGG-16 Resnet-34 Resnet-50 ViT-Small ViT-Base Bert-MRPC Bert-SST2 Geomean<br>Fig. 12: Speedup results normalized to Stripes (higher is better).<br>Off-chip Memory On-chip Compute<br>1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>VGG-16 ResNet-34 ResNet-50 ViT-Small ViT-Base Bert-MRPC Bert-SST2 Geomean<br>×<br>× 3.03<br>2.48<br>×<br>×× × × 1.83<br>1.49 1.52 ×<br>1.33<br>Speedup 1.20<br>0.63×<br>0.57× 0.59× ×<br>0.45× 0.52× 0.47× 0.41<br>Norm. Energy<br>SparTen ANT Stripes Pragmatic Bitlet BitWave BitVert (cons) BitVert (mod) SparTen ANT Stripes Pragmatic Bitlet BitWave BitVert (cons) BitVert (mod) SparTen ANT Stripes Pragmatic Bitlet BitWave BitVert (cons) BitVert (mod) SparTen ANT Stripes Pragmatic Bitlet BitWave BitVert (cons) BitVert (mod) SparTen ANT Stripes Pragmatic Bitlet BitWave BitVert (cons) BitVert (mod) SparTen ANT Stripes Pragmatic Bitlet BitWave BitVert (cons) BitVert (mod) SparTen ANT Stripes Pragmatic Bitlet BitWave BitVert (cons) BitVert (mod) SparTen ANT Stripes Pragmatic Bitlet BitWave BitVert (cons) BitVert (mod)<br>**----- End of picture text -----**<br>


Fig. 13: Energy consumption breakdown normalized to SparTen (lower is better). 

**Comparison against ANT** We compare the accuracy between moderate binary pruning and ANT [16]. As shown in Table II, BBS outperforms ANT in terms of both accuracy and effective weight bit width. While ANT uses adaptive datatypes for lowbit quantization, it cannot take the advantage of inherent bitlevel redundancy. On the other hand, the binary pruning fully exploits the bit-level sparsity to best preserve the original 8-bit weight distribution, resulting in minimal accuracy degradation. 

**Comparison against PTQ Works** We compare the accuracy loss between BBS and state-of-the-art PTQ works, including Microscaling [36] and NoisyQuant [24], on vision transformers. We apply 6-bit weight quantization using the two PTQ methods while maintaining activation to 8-bit. Table III shows that the moderate binary pruning outperforms NoisyQuant with lower memory footprint. Moreover, the conservative binary pruning has much better accuracy than Microscaling at similar bit width. Miscroscaling also has an 8-bit metadata, which represents the shared exponent for a group of 32 weights. However, the exponent is determined by the largest value in every group, which forces small values to become zero due to insufficient operand precision to store the aligned mantissa. On the other hand, BBS exploits bit-level redundancy to better preserve the statistical characteristics of 

|Model|**BBS (mod)**|ANT [16]|
|---|---|---|
|VGG-16|**0.2% (4.32 bits)**|0.68% (6 bits)|
|ResNet-50|**0.23% (4.79 bits)**|0.89% (6 bits)|



TABLE II. Comparison of accuracy loss and weight bit width between BBS and 6-bit ANT without fine-tuning. 

uncompressed weight, thereby achieving higher accuracy. 

## _C. Accelerator Performance and Energy_ 

**Performance** Fig. 12 presents the accelerator performance normalized to that of Stripes. On average, _BitVert_ with conservative and moderate binary pruning achieves 2 _._ 48 _×_ and 3 _._ 03 _×_ speedup compared to Stripes, respectively. These speedups are attributed to exploiting both balanced BBS and binary pruning for abundant bit skipping and reduced memory access. Despite leveraging two-sided value sparsity, SparTen demonstrates limited performance on transformer-based models due to the lack of weight value sparsity in 8-bit models and nearly-dense activations from non-ReLU functions. ANT only explores reduced value precision but not fine-grained bit-level sparsity, leading to 1 _._ 63 _×_ and 1 _._ 97 _×_ lower speedup than _BitVert_ at conservative and moderate pruning, respectively. While Pragmatic and Bitlet utilize variable degrees of bit-level sparsity, they suffer from workload imbalance and lack of exploration in further compressing DNNs below 8-bit. This explains why _BitVert_ outperforms Pragmatic and Bitlet by 1 _._ 86 – 2 _._ 53 _×_ across all benchmarks. Although BitWave exploits structured 

||ViT-Small|ViT-Small|ViT-Base|ViT-Base|
|---|---|---|---|---|
||∆Acc _↓_|Bits|∆Acc _↓_|Bits|
|Microscaling [36]|2_._49%|6.25|0_._33%|6.25|
|NoisyQuant [24]|2_._08%|6|0_._64%|6|
|BBS (cons)|0_._75%|6.33|0_._05%|6.25|
|BBS (mod)|0_._96%|5.19|0_._39%|5.07|



TABLE III. Comparison of accuracy loss and weight bit width between BBS, Microscaling and NoisyQuant. 

560 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:15:39 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [227 x 103] intentionally omitted <==**

Fig. 14: Normalized speedup on ResNet-50 and Bert-MRPC with increasing number of PE columns ( _i.e._ , processing more weight groups in parallel). 

bit-column pruning to achieve better performance, its moderate pruning results in unacceptable accuracy loss ( _>_ 1%) on many DNNs such as ViT-small and Bert-MRPC. Therefore, it has to reduce the degree of pruning for improved accuracy while sacrificing performance. Overall, _BitVert_ provides the best accuracy-performance trade-offs, with up to 1 _._ 98 _×_ speedup over BitWave. 

**Energy Consumption** Fig. 13 presents the normalized energy breakdown of different accelerators. where the on-chip compute energy includes both buffer and core energy. SparTen demonstrates the poorest energy efficiency primarily due to its substantial overhead from the sparse bitmask encoding (12 _._ 5% at 8-bit precision) and the expensive hardware required to exploit sparsity. This overhead is particularly pronounced in 8-bit DNNs, where value sparsity is inherently scarce. As a result, SparTen consumes 2 _._ 13 _×_ and 2 _._ 44 _×_ higher energy than _BitVert_ with conservative and moderate pruning, respectively. Although ANT is able to quantize both activations and weights, it dissipates higher energy than _BitVert_ with moderate pruning due to the complicated hardware to support custom data types. Owing to the balanced BBSskipping and substantial reduction in model size, _BitVert_ with moderate pruning achieves an average energy reduction of 1 _._ 39 _×_ , 1 _._ 43 _×_ , 1 _._ 54 _×_ , and 1 _._ 27 _×_ over Stripes, Pragmatic, Bitlet, and BitWave, respectively. 

## _D. Analysis of Load Imbalance_ 

_BitVert_ can leverage the structured BBS for improved load balance. Fig. 14 demonstrates this with the performance on ResNet-50 and Bert-MRPC with respect to different number of PE columns, where every PE column processes a different weight group. When there are more PE columns, Pragmatic and Bitlet exhibit a noticeable drop in speedup over Stripes that does not exploit bit sparsity. For instance, when the number of PE columns increases from 2 to 32, the speedup of Bitlet on Bert-MRPC drops from 1 _._ 63 _×_ to 1 _._ 35 _×_ . This is because that processing more weight groups in parallel exacerbates the load imbalance across PE columns, and the performance is bottlenecked by the weight group with the lowest bit sparsity. In contrast, the structured bit sparsity allow BitWave and _BitVert_ to efficiently scale the performance, thus maintaining nearly constant speedup over Stripes. Moreover, _BitVert_ always 

**==> picture [203 x 112] intentionally omitted <==**

Fig. 15: Breakdown of execution cycles w.r.t. the number of PE columns. 

achieves the highest performance thanks to the binary pruning that can induce higher BBS with negligible accuracy loss. 

Fig. 15 further details the breakdown of execution time with respect to the number of PE columns to highlight its impact on load balance. Since one PE contains many bit-serial multipliers, intra-PE stall can be caused by a multiplier that needs to process more effectual bits. On the other hand, the inter-PE stall arises from variance in bit sparsity across different weight groups. As the number of PE columns increases, Pragmatic and Bitlet experience higher intra-PE and inter-PE loss, which explains their lower resulting speedup. BitWave only exploits coarse-grained bit-column sparsity that has much lower occurrence than fine-grained BBS. Therefore, it shows lower PE utilization than _BitVert_ . Furthermore, _BitVert_ has minimal inter-PE stall due to the more balanced distribution of BBS across different weight groups, thereby achieving superior performance over other bit-serial accelerators. 

## _E. PE Design Space Exploration_ 

Recall from Section IV-A that the sub-group size within the _BitVert_ PE offers a trade-off between area and power. A smaller sub-group has lower mux cost, but increases the number of subtractors. Furthermore, by exploiting the structured nature of BBS and its encoding scheme, we are able to further reduce the PE area by using compact mux and a smaller BBS multiplier. Hence, we conduct a PE design space exploration to evaluate the optimal group size and the proposed optimizations. As shown in Table IV, a sub-group size of 16 without optimization incurs a significant area overhead of 38 _._ 2% compared to the optimized design. In the end, a sub-group size of 8 with the proposed PE optimization offers the best trade-off between area and power, which is therefore adopted in our _BitVert_ accelerator. 

|Sub-group<br>Size<br>A|Without Optimization|With Optimization|
|---|---|---|
||rea (_um_2)<br>Power (_mW_)|Area (_um_2)<br>Power (_mW_)|
|16|1342.3<br>0.61|971.5<br>0.53|
|8|896.6<br>0.49|739.6<br>0.45|
|4|878.7<br>0.51|786.5<br>0.47|



TABLE IV. PE area and power of _BitVert_ with different sub-group sizes before and after applying our circuit optimizations. 

561 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:15:39 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [236 x 123] intentionally omitted <==**

**----- Start of picture text -----**<br>
Bitlet PTQ ANT BitWave BitVert<br>1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>0<br>0.2 0.4 0.6 0.8 1.0<br>ResNet-50 Normalized EDP<br>Accuracy loss (%) better<br>**----- End of picture text -----**<br>


**==> picture [107 x 108] intentionally omitted <==**

**==> picture [95 x 108] intentionally omitted <==**

Fig. 17: Comparison between BBS and Olive on compressing Llama-3-8B weights. The accuracy metric is perplexity, **lower is better** . 

Fig. 16: EDP-acccuracy loss pareto frontier for ResNet50. 

## _F. PE Area and Power Comparison_ 

The _BitVert_ accelerator adopts an area- and energy-efficient PE with low overhead to support BBS. We compare the PE design of _BitVert_ and other bit-serial accelerators, with all PEs containing 8 bit-serial multipliers at 800 MHz target frequency. Table V summarizes the area and power of different PEs. Bitlet experiences the highest area and power consumption due to significant overhead ( _e.g._ , a 64-1 mux before every bit-serial multiplier) for zero bit skipping. Pragmatic needs a variable shifter to align the bit significance, leading to a larger bitserial multiplier and non-trivial overhead. BitWave requires 2’s complementer to support sign-magnitude arithmetic, resulting in 1 _._ 32 _×_ larger area and 1 _._ 4 _×_ power than Stripes. Moreover, since BitWave can only leverage coarse-grained bit-column sparsity, the potential performance improvement is limited. The proposed _BitVert_ enjoys the optimal trade-off between performance and hardware cost. Its PE occupies 1 _._ 39 _×_ area and consumes 1 _._ 22 _×_ power compared to Stripes, yet is able to exploit 50% balanced BBS and binary pruning for efficient bit skipping and model compression, respectively. Since BBS naturally exists in a bit-vector with arbitrary length and does not depend on the operand precision, it provides a promising solution for future bit-serial computing paradigm. 

## _G. Accuracy-Efficiency Trade-offs_ 

The proposed binary pruning and _BitVert_ can offer good trade-offs between accuracy and efficiency. To demonstrate this, we conduct design-space exploration on ResNet-50 with different pruning ratios. We compare the relationship between energy-delay product (EDP) and accuracy loss of _BitVert_ and previous works, including Bitlet, BitWave, ANT and conven- 

|Accelerator|PE Area (_um_2)|PE Power<br>(_mW_)|
|---|---|---|
||Multiplier<br>Others<br>Total<br>Ratio||
|Stripes [19]|286.3<br>246.5<br>532.8<br>1_×_|0.37|
|Pragmatic [1]|319.2<br>603.9<br>923.1<br>1_._73_×_|0.51|
|Bitlet [26]|223.2<br>1442.4<br>1665.6<br>3_._13_×_|0.57|
|BitWave [39]|286.3<br>416.1<br>702.4<br>1_._32_×_|0.49|
|**BitVert (ours)**|**332.4**<br>**407.2**<br>**739.6**<br>**1.39**_×_|**0.45**|



TABLE V. PE area and power of BitVert and prior bit-serial accelerators under 28 nm technology and 800 MHz frequency. 

|Accelerator|Area<br>(_um_2)|Power<br>(_mW_)|Norm.<br>Perf|Norm.<br>Perf / Area|
|---|---|---|---|---|
|Olive [15]|291.6|0.18|1_×_|1_×_|
|BitVert (mod)|739.6|0.45|4_×_|1.58_×_|



TABLE VI. Comparison between Olive and _BitVert_ PEs. 

tional PTQ. As shown in Fig. 16, the lower left region indicates a good trade-off between accuracy and EDP. Although BitWave and ANT propose different algorithm-hardware codesign approaches for DNN compression and acceleration, they fail to preserve the original value distribution of the baseline model and do not efficiently leverage the balanced bit sparsity that inherently appears in DNNs. In contrast, binary pruning is able to preserve all quantization levels of the original DNN. Combining with BBS and efficient hardware design, _BitVert_ is able to always sit on the Pareto frontier. 

## _H. Applicability to Large Language Models_ 

Large language models (LLMs) have achieved great success in generative tasks [40], [47]. We compare BBS with a recent PTQ work Olive [15] for LLM weight compression. We evaluate a state-of-the-art LLM, Llama-3-8B [29] on Wikitext [28] and C4 [8] datasets. For BBS, we apply conservative and moderate binary pruning to _all_ weight channels with a group size of 32, resulting in an effective weight precision of 6.25 and 4.25 bits, respectively. Fig. 17 shows the accuracy impact of different compression methods. The moderate BBS pruning achieves better perplexity than Olive with a similar memory footprint (4.25 vs. 4 bits), while the conservative BBS pruning has little perplexity loss compared to the FP32 baseline. To compare the hardware efficiency, we synthesize the Olive PE for 4-bit weight and 8-bit activation. Table VI shows that the proposed _BitVert_ PE with moderate binary pruning can achieve 1 _._ 58 _×_ better performance per area compared to Olive. The benefits of _BitVert_ are twofold. First, Olive adopts separate datatypes for normal and outlier values, where the latter has a much wider numerical range. Therefore, the Olive PE requires a larger multiplier than fixed-point PE to accommodate outliers. Second, the _BitVert_ PE exploits BBS to efficiently compute 16 multiplications in 4 cycles under moderate pruning, while the Olive PE does not leverage bit sparsity and only computes one multiplication per cycle. 

562 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:15:39 UTC from IEEE Xplore.  Restrictions apply. 

## VI. CONCLUSION 

In this paper, we introduce BBS, a new concept to exploit bit-level sparsity in a symmetrical way to prune either zerobits or one-bits. BBS pushes the limit of post-training DNN compression to a new state-of-the-art through binary pruning, a data-free optimization that generates bi-directional sparse bit columns inside DNN weights while maximally preserving the statistical characteristics of the original uncompressed model. As a result, the proposed binary pruning technique achieves much higher accuracy compared to previous bit-sparsity-aware pruning methods. On top of the algorithmic innovation, we design a bit-serial accelerators named _BitVert_ with an areaand power-efficient PE to fully mine the potential of BBS. Compared to prior DNN accelerators, _BitVert_ achieves up to 3 _._ 03 _×_ speedup and 2 _._ 44 _×_ energy saving, while having negligible accuracy degradation on both vision and language models with large-scale benchmark datasets. 

## ACKNOWLEDGMENT 

This project is supported in part by Intel Corporation and the Center for the Co-Design of Cognitive Systems (CoCoSys) in JUMP 2.0, an SRC Program sponsored by DARPA. We would like to thank Mahesh Iyer, Grace Zgheib, Sergey Gribok, Ahmed AbouElhamayed, Zhewen Yu, Marta Andronic, and the anonymous reviewers for their constructive feedback. We also thank Man Shi for the helpful discussion about BitWave. The code for BBS binary pruning can be found at https://github.com/yc2367/BBS-MICRO.git. 

## REFERENCES 

- [1] J. Albericio, A. Delmas, P. Judd, S. Sharify, G. O’Leary, R. Genov, and A. Moshovos, “Bit-Pragmatic deep neural network computing,” _50th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2017. 

- [2] H. An, Y. Chen, Z. Fan, Q. Zhang, P. Abillama, H.-S. Kim, D. Blaauw, and D. Sylvester, “An 8.09tops/w neural engine leveraging bit-sparsified sign-magnitude multiplications and dual adder trees,” _IEEE International Solid- State Circuits Conference (ISSCC)_ , pp. 422–424, 2023. 

- [3] T. Andrulis, J. S. Emer, and V. Sze, “RAELLA: Reforming the arithmetic for efficient, low-resolution, and low-loss analog pim: No retraining required!” _Proceedings of the 50th Annual International Symposium on Computer Architecture (ISCA)_ , 2023. 

- [4] R. Balasubramonian, A. B. Kahng, N. Muralimanohar, A. Shafiee, and V. Srinivas, “CACTI 7: New tools for interconnect exploration in innovative off-chip memories,” _ACM Trans. Archit. Code Optim._ , vol. 14, no. 2, June 2017. 

- [5] Y. Cai, Z. Yao, Z. Dong, A. Gholami, M. W. Mahoney, and K. Keutzer, “Zeroq: A novel zero shot quantization framework,” _arXiv preprint arXiv:2001.00281_ , 2020. 

- [6] C. Deng, Y. Sui, S. Liao, X. Qian, and B. Yuan, “GoSPA: An energyefficient high-performance globally optimized sparse convolutional neural network accelerator,” _2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)_ , 2021. 

- [7] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” _North American Chapter of the Association for Computational Linguistics_ , 2019. 

- [8] J. Dodge, A. Marasovic, G. Ilharco, D. Groeneveld, M. Mitchell, and M. Gardner, “Documenting large webtext corpora: A case study on the colossal clean crawled corpus,” in _Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2021. 

- [9] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” _arXiv preprint abs/2010.11929_ , 2020. 

- [10] S. K. Esser, J. L. McKinstry, D. Bablani, R. Appuswamy, and D. S. Modha, “Learned step size quantization,” _arXiv preprint arXiv:1902.08153_ , 2019. 

- [11] A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, and K. Keutzer, “A survey of quantization methods for efficient neural network inference,” _arXiv preprint arXiv:2103.13630_ , 2021. 

- [12] A. Gholami, Z. Yao, S. Kim, C. Hooper, M. W. Mahoney, and K. Keutzer, “AI and memory wall,” _IEEE Micro_ , 2024. 

- [13] A. Gondimalla, N. Chesnut, M. Thottethodi, and T. N. Vijaykumar, “SparTen: A sparse tensor accelerator for convolutional neural networks,” _Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2019. 

- [14] A. Gondimalla, M. Thottethodi, and T. N. Vijaykumar, “Eureka: Efficient tensor cores for one-sided unstructured sparsity in dnn inference,” _2023 56th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2023. 

- [15] C. Guo, J. Tang, W. Hu, J. Leng, C. Zhang, F. Yang, Y.-B. Liu, M. Guo, and Y. Zhu, “OliVe: Accelerating large language models via hardwarefriendly outlier-victim pair quantization,” _50th ACM/IEEE International Symposium on Computer Architecture (ISCA)_ , 2023. 

- [16] C. Guo, C. Zhang, J. Leng, Z. Liu, F. Yang, Y.-B. Liu, M. Guo, and Y. Zhu, “ANT: Exploiting adaptive numerical data type for low-bit deep neural network quantization,” _55th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2022. 

- [17] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” _arXiv preprint arXiv:1503.02531_ , 2015. 

- [18] D. Im, G. Park, Z. Li, J. Ryu, and H.-J. Yoo, “Sibia: Signed bit-slice architecture for dense dnn acceleration with slice-level sparsity exploitation,” _IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ , 2023. 

- [19] P. Judd, J. Albericio, and A. Moshovos, “Stripes: Bit-serial deep neural network computing,” _49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2016. 

- [20] A. D. Lascorz, P. Judd, D. M. Stuart, Z. Poulos, M. Mahmoud, S. Sharify, M. Nikolic, K. Siu, and A. Moshovos, “Bit-Tactical: A software/hardware approach to exploiting value and bit sparsity in neural networks,” _Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)_ , 2019. 

- [21] N. Lee, T. Ajanthan, and P. H. S. Torr, “SNIP: Single-shot network pruning based on connection sensitivity,” _arXiv preprint arXiv:1810.02340_ , 2019. 

- [22] S.-J. Li, Z. Yang, D. Reddy, A. Srivastava, and B. Jacob, “DRAMsim3: A cycle-accurate, thermal-capable dram simulator,” _IEEE Computer Architecture Letters_ , vol. 19, pp. 106–109, 2020. 

- [23] F. Liu, W. Zhao, Z. He, Z. Wang, Y. Zhao, Y. Chen, and L. Jiang, “BitTransformer: Transforming bit-level sparsity into higher preformance in reram-based accelerator,” _IEEE/ACM International Conference On Computer Aided Design (ICCAD)_ , 2021. 

- [24] Y. Liu, H. Yang, Z. Dong, K. Keutzer, L. Du, and S. Zhang, “NoisyQuant: Noisy bias-enhanced post-training activation quantization for vision transformers,” _arXiv preprint arXiv:2211.16056_ , 2023. 

- [25] Z. Liu, Y. Wang, K. Han, S. Ma, and W. Gao, “Post-training quantization for vision transformer,” _arXiv preprint arXiv:2106.14156_ , 2021. 

- [26] H. Lu, L. Chang, C. Li, Z. Zhu, S. Lu, Y. Liu, and M. Zhang, “Distilling bit-level sparsity parallelism for general purpose deep learning acceleration,” _54th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2021. 

- [27] E. Meller, A. Finkelstein, U. Almog, and M. Grobman, “Same, same but different - recovering neural network quantization error through weight factorization,” _arXiv preprint arXiv:1902.01917_ , 2019. 

- [28] S. Merity, C. Xiong, J. Bradbury, and R. Socher, “Pointer sentinel mixture models,” _arXiv preprint arXiv:1609.07843_ , 2016. 

- [29] Meta, “Meta llama 3.” [Online]. Available: https://github.com/metallama/llama3 

- [30] A. Mishra, J. A. Latorre, J. Pool, D. Stosic, D. Stosic, G. Venkatesh, C. Yu, and P. Micikevicius, “Accelerating sparse deep neural networks,” _arXiv preprint arXiv:2104.08378_ , 2021. 

563 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:15:39 UTC from IEEE Xplore.  Restrictions apply. 

- [31] P. Molchanov, S. Tyree, T. Karras, T. Aila, and J. Kautz, “Pruning convolutional neural networks for resource efficient transfer learning,” _International Conference on Learning Representations,_ , 2017. 

- [32] M. Nagel, M. van Baalen, T. Blankevoort, and M. Welling, “Data-free quantization through weight equalization and bias correction,” _arXiv preprint arXiv:1906.04721_ , 2019. 

- [33] NVIDIA, “Tensorrt: A c++ library for high performance inference on nvidia gpus and deep learning accelerators.” [Online]. Available: https://github.com/NVIDIA/TensorRT 

- [34] E. Park, D. Kim, and S. Yoo, “Energy-efficient neural network accelerator based on outlier-aware low-precision computation,” _ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA)_ , 2018. 

- [35] S. Qu, B. Li, Y. Wang, and L. Zhang, “ASBP: Automatic structured bit-pruning for rram-based nn accelerator,” _58th ACM/IEEE Design Automation Conference (DAC)_ , 2021. 

- [36] B. D. Rouhani, R. Zhao, V. Elango, R. Shafipour, M. Hall, M. Mesmakhosroshahi, A. More, L. Melnick, M. Golub, G. Varatkar, L. Shao, G. Kolhe, D. Melts, J. Klar, R. L’Heureux, M. Perry, D. Burger, E. S. Chung, Z. Deng, S. Naghshineh, J. Park, and M. Naumov, “With shared microexponents, a little shifting goes a long way,” _ACM/IEEE 50th Annual International Symposium on Computer Architecture (ISCA)_ , 2023. 

- [37] S. Sharify, A. D. Lascorz, M. Mahmoud, M. Nikolic, K. Siu, D. M. Stuart, Z. Poulos, and A. Moshovos, “Laconic deep learning inference acceleration,” _ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA)_ , 2019. 

- [38] H. Sharma, J. Park, N. Suda, L. Lai, B. Chau, J. K. Kim, V. Chandra, and H. Esmaeilzadeh, “Bit Fusion: Bit-Level Dynamically Composable Architecture for Accelerating Deep Neural Network,” in _45th ACM/IEEE International Symposium on Computer Architecture (ISCA)_ , 2018. 

- [39] M. Shi, V. Jain, A. Joseph, M. Meijer, and M. Verhelst, “BitWave: Exploiting column-based bit-level sparsity for deep learning acceleration,” _Proceedings of the 30th IEEE International Symposium on HighPerformance Computer Architecture (HPCA)_ , 2024. 

- [40] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample, “LLaMA: Open and efficient foundation language models,” _arXiv preprint arXiv:2302.13971_ , 2023. 

- [41] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman, “GLUE: A multi-task benchmark and analysis platform for natural language understanding,” _arXiv preprint arXiv:1804.07461_ , 2018. 

- [42] Y. Wang, C. Zhang, Z. Xie, C. Guo, Y. Liu, and J. Leng, “Dualside sparse tensor core,” _2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)_ , 2021. 

- [43] Y. N. Wu, P.-A. Tsai, S. Muralidharan, A. Parashar, V. Sze, and J. S. Emer, “HighLight: Efficient and flexible dnn acceleration with hierarchical structured sparsity,” _56th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2023. 

- [44] G. Xiao, J. Lin, M. Seznec, J. Demouth, and S. Han, “SmoothQuant: Accurate and efficient post-training quantization for large language models,” _arXiv preprint arXiv:2211.10438_ , 2022. 

- [45] Z. Yuan, C. Xue, Y. Chen, Q. Wu, and G. Sun, “PTQ4ViT: Post-training quantization framework for vision transformers with twin uniform quantization,” _arXiv preprint arXiv:2111.12293_ , 2022. 

- [46] A. H. Zadeh, I. Edo, O. M. Awad, and A. Moshovos, “GOBO: Quantizing attention-based nlp models for low latency and energy efficient inference,” _53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ , 2020. 

- [47] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. T. Diab, X. Li, X. V. Lin, T. Mihaylov, M. Ott, S. Shleifer, K. Shuster, D. Simig, P. S. Koura, A. Sridhar, T. Wang, and L. Zettlemoyer, “OPT: Open pre-trained transformer language models,” _arXiv preprint arXiv:2205.01068_ , 2022. 

564 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:15:39 UTC from IEEE Xplore.  Restrictions apply. 

