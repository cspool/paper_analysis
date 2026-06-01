# GPU Large Data Access

ref：METHOD AND APPARATUS FOR EFFICIENT ACCESS TO MULTIDIMENSIONAL DATA STRUCTURES AND/OR OTHER LARGE DATA BLOCKS

## Fig1、2

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 1
![image.png](GPU%20Large%20Data%20Access/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Large%20Data%20Access/image%201.png)

## Fig3、4、5

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 3A
![image.png](GPU%20Large%20Data%20Access/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 3B
![image.png](GPU%20Large%20Data%20Access/image%203.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> FIG. 4A
> 
> ![](_page_0_Picture_2.jpeg)
> 
> FIG. 4B
![image.png](GPU%20Large%20Data%20Access/image%204.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 5A
> 
> ![](_page_0_Figure_2.jpeg)
> 
> FIG. 5B
![image.png](GPU%20Large%20Data%20Access/image%205.png)

## Fig6、7、8

TMAU：tensor memory load，每个元素有对应地址。

GEMM for Conv的例子，warps。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 6
![image.png](GPU%20Large%20Data%20Access/image%206.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 7A
> 
> boxSize[1] =
![image.png](GPU%20Large%20Data%20Access/image%207.png)

> **[图片提取文字 (image.png)]:**
> ```
> anchorGlobalAddr[0] = coord[0] * elementSizeInBytes();
> for (i = 1; i < 5; i++){
>    anchorGlobalAddr[i] = coord[i] * tensorDescriptor.tensorStride[i-1];
>    tensorDescriptor.tensorStride[i-1] *= tensorDescriptor.traversalStride[i];
> sharedAddr = initSharedAdd();
> baseGlobalAddr[4] = anchorGlobalAddr[4];
> for (c4 = blockStart4; c4 < blockStart4 + tensorDescriptor, boxSize[4];c4 += tensorDescriptor.traversalStride[4]){
>    baseGlobalAddr[3] = baseGlobalAddr[4] + anchorGlobalAddress[3];
>   for (c3 = blockStart3; c3 < blockStart3 + tensorDescriptor.boxSize[3]; c3 += tensorDescriptor.traversalStride[3]){
>       baseGlobalAddr[2] = baseGlobalAddr[3] + anchorGlobalAddress[2];
>      for (c2 = blockStart2; c2 < blockStart2 + tensorDescriptor.boxSize[2];c2 += tensorDescriptor.traversalStride[2]){
>         baseGlobalAddr[1] = baseGlobalAddr[2] + anchorGlobalAddress[1];
>         for (c1 = blockStart1; c1< blockStart1 + tensorDescriptor.boxSize[1];c1 += tensorDescriptor.traversalStride[1]){
>                  baseGlobalAddress[0] = baseGlobalAddr[1] + anchorGlobalAddress[0];
>                  int globalAddr = baseGlobalAddr[0];
>                  for (c0 = blockStart0; c0 < blockStart0 + tensorDescriptor.boxSize[0]; c0 += tensorDescriptor.traversalStride[0]){
>                       processElement(c0, c1, c2, c3, c4, sharedAddr, globalAddr);
>                       globalAddr += elementSizeInBytes();
>                       sharedAddr += sharedAddrIncrement();
>                    // c0
>                  baseGlobalAddr[1] += tensorDescriptor.tensorStride[0]:
>                                                                                      // advance global address to next slice
>           } // c1
>           baseGlobalAddr[2] += tensorDescriptor.tensorStride[1];
>                                                                       // advance global address to next slice
>         // c2
>       baseGlobalAddr[3] += tensorDescriptor.tensorStride[2];
>                                                                        // advance global address to next slice
>   } // c3
> baseGlobalAddr[4] += tensorDescriptor.tensorStride[3];
>                                                                      // advance global address to next slice
>    // c4
> ```
> 
> FIG. 7B
![image.png](GPU%20Large%20Data%20Access/image%208.png)

> **[图片提取文字 (image.png)]:**
> ```
> // define tensor descriptors
> tensor activationTensor = tensorDescriptor(activationGlobalAddr, cTensorSize, hTensorSize, mTensorSize, nTensorSize,
>                                                                          cBlockSize, hBlockSize, wBlockSize, nTensorSize);
> _tensor weightTensor = tensorDescriptor( weightGlobalAddr, cTensorSize, sTensorSize, rTensorSize, kTensorSize,
>                                                                          cBlockSize, sBlockSize, rBlockSize, kTensorSize);
> tensor outputTensor = tensorDescriptor(outputGlobalAddr, wTensorSize, hTensorSize, nTensorSize, kTensorSize,
>                                                                          wBlockSize, hBlockSize, nBlockSize, kTensorSize);
> // iterate through tensor space and compute
> for (int h = 0; C < hTensorSize; h += hBlockSize) {
>                                                                          // iterate through tensor vertically
>    for (int w = 0; w \le wTensorSize; w += wBlockSize) {
>                                                                            // iterate through tensor horizontally
>        for (int c = 0: c < cTensorSize: c += cBlockSize) {
>                                                                            // iterate through tensor channels
>            for (int r = -filterHeight/2, R \le -filterHeight/2, r++){
>                                                                            // iterate through filter vertically
>                for (int s = -filterWidth/2, s \le filterWidth/2, s++){
>                                                                            // iterate through filter horizontally
>                            // TMA loads block from activation tensor: global -> shared memory
>                    tensorBlockLoad(activationTensor, activationSMEMaddr, c, w + s, h + r, n);
>                            // TMA loads block from weight tensor: global -> shared memory
>                    tensorBlockLoad(weightTensor, weightSMEMaddr, c, s, r, k);
>                            // GMMA math. Write results to shared memory
>                    computeGEMM( activationSMEMaddr, weightSMEMaddr, outputSMEMaddr);
>        // TMA stores results to output tensor: shared -> global memory
>        tensorBlockStore(outputTensor,outputSMEMaddr, w. h, n. k);
> ```
> 
> FIG. 7C
![image.png](GPU%20Large%20Data%20Access/image%209.png)

conv tiled load：读取张量的NHW64，C8，面向没有数据重叠的情况（s大于等于k）。

im2col load mode在加载时完成img2col（滑窗地址计算更复杂），支持有重叠数据的Conv  。

> **[图片提取文字 (image.png)]:**
> kernels based on implicit GEMM. If im2col mode is selected, then TMAU does image-to-column transformation when it loads tensor blocks from global memory. This adds extra complexity to the tensor traversal algorithm. [0106] In the tiled mode, the tensor parameter boxSize[] uniquely defines boundingBox size in the tensor space that holds all the elements that the TMAU is supposed to load in response to an instruction from the SM. Each element of the boxSize[] specifies boundingBox size along a corresponding dimension: boundingBox[i]=boxSize[i]. The coordinates specified in a TMAU memory access request from the SM uniquely define the location of the boundingBox in the tensor space. [0107] In the im2col mode, the boundingBox size and location are defined differently. The number of boundingBox dimensions is one less than the tensor dimensionality in the tensor descriptor. The boxSize[] is not used in this mode, and instead there are alternative parameters in the tensor descriptor to support the im2col mode. The alternative parameters include the following: rangeNDHW, rangeC, boxBaseCornerDHW, boxFarCornerDHW. boxBaseCornerDHW and boxFarCornerDHW define boundingBox size and location in DHW (Depth, Height, Width) space. The boxBaseCornerDHW specifies initial coordinates of the boundingBox origin which is box upper left corner. The boxFarCornerDHW specifies initial location of the opposite right bottom corner. The corners' locations are defined as signed offsets from the corresponding tensor corners. Therefore, the bounding box corners could be
> 
> specified both inside and outside of the tensor boundaries.
> 
> [0105] The im2col mode is primarily used in convolution
![image.png](GPU%20Large%20Data%20Access/image%2010.png)

> **[图片提取文字 (image.png)]:**
> the boxBaseCorner {D,H,W}, boxFarCorner {D,H,W} settings. This example shows that many types of borders may be used in the data structures, and in the im2col mode, quantization can be avoided. [0111] In the tiled mode, the number of elements to load depends on the boxSize parameters. When the TMAU traverses a particular dimension, it uses the corresponding value from the boxSize[] to determine how many elements to load. In the im2col mode rangeNDHW is used to determine how many elements to load along NDHW dimensions and rangeC for the dimension C. A single TMAU request may require the TMAU to traverse multiple images from a
> 
> batch (N dimension) in order to load a requested number of
> 
> elements. When TMAU switches from the current image to
> 
> [0110] FIG. 8A illustrates how boundingBox depends on
![image.png](GPU%20Large%20Data%20Access/image%2011.png)

> **[图片提取文字 (image.png)]:**
> next during traversal of multiple images, it may skip channels that are outside the range defined by rangeC parameter. [0112] In the tiled mode, the TMAU request coordinates specify boundingBox location (origin) in the tensor space. In
> 
> im2col mode, coordinates along C and N dimensions are
> 
> used similar to the tiled mode; however, coordinates along
> 
> W, H, D dimensions specify the base location of the convolution filter (upper left corner) in the tensor space. For correct processing, the TMAU requires that the base location of the filter is always be defined within the boundingBox. In addition, coordinate offsets for these dimensions have to be specified in the TMAU request. The offsets allows the position of the block to be specified relative to the tensor, and therefore using only a minimal number of bytes. The offsets are added to the filter base location coordinates to determine starting locations in the tensor space from where the load operation must be initiated. The same offsets are used to position boundingBox relative to the initial coordinates specified in boxBaseCornerDHW. The offsets are
> 
> for the bounding box coordinates precision.
> 
> [0113] In some embodiments, all offsets are packed in 16 bits within a single register. The number of offsets depends on the tensor dimensionality; therefore, the precision may vary accordingly. In the typical convolution kernel once the filter base is calculated it could be reused for multiple TMAU requests with different coordinate offsets. The number of reuses depends on the convolution filter size. For
> 
> applied to subset of the coordinates based on the table
> 
> defined above. The offsets are defined as unsigned integer
> 
> with variable precision. The precision depends on the tensor
> 
> dimensionality and chosen based on the earlier justification
> 
> same filter base location.
> 
> [0114] For the interleaved layouts, the C coordinate must be specified in terms of channel slices rather than individual channels. This applies to both tiled and im2col modes.
> 
> example, for a 3×3 filter, nine requests are issued for the
![image.png](GPU%20Large%20Data%20Access/image%2012.png)

深色代表加载数据，不同H/W表示不同Base的load request。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 8A
![image.png](GPU%20Large%20Data%20Access/image%2013.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_1.jpeg)
> 
> FIG. 8B
![image.png](GPU%20Large%20Data%20Access/image%2014.png)

conv load quantization：加载需要将计算尺寸对齐颗粒度，浪费加载数据。

> **[图片提取文字 (image.png)]:**
> Image 7
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> ![](_page_0_Figure_4.jpeg)
> 
> FIG. 8C
![image.png](GPU%20Large%20Data%20Access/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 8D
> 
> ![](_page_0_Figure_2.jpeg)
> 
> FIG. 8E
![image.png](GPU%20Large%20Data%20Access/image%2016.png)

traversalStride=2表示每次访问数据相互间隔2个pixel，im2col load mode的加载次数依赖traversalStride，加载元素数和stride无关，说明低效访存。

> **[图片提取文字 (image.png)]:**
> ceil(boundingBox{D,H, W}/traversalStride{D,H,W}).
> 
> [0122] The tensor descriptor traversalStride parameter
![image.png](GPU%20Large%20Data%20Access/image%2017.png)

> **[图片提取文字 (image.png)]:**
> im2col mode. A 3×3 convolution filter is applied to NHWC tensor (64×14×9×64) with traversalStride equal two. Each request loads 32 elements along N, H, W dimensions, and 16 elements along C. The tensor descriptor parameters are set up as the following: tensorSize[0]=64; tensorSize[1]=9; tensorSize[2]=14; tensorSize[4]=64; traversalStride=2; rangeNDHW=32; rangeC=16; boxBaseCornerW=-1; boxBaseCornerH=-1; boxFarCornerW=-1; boxFarCornerH=-1. FIG. 8B illustrates processing for the requests with coordinates (7, 7, 5, 0) and different coordinate offset values: (0, 0), (1, 1), (2, 2). Note that in this example pixels are loaded from the top row of the boundingBox, but not from the bottom row. They are also loaded from both first and last columns.
> 
> [0123] FIG. 8F illustrates traversalStride handling in
![image.png](GPU%20Large%20Data%20Access/image%2018.png)

> **[图片提取文字 (image.png)]:**
> Image 7
> 
> ![](_page_0_Figure_1.jpeg)
> 
> FIG. 8F
![image.png](GPU%20Large%20Data%20Access/image%2019.png)

> **[图片提取文字 (image.png)]:**
> Image 7
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Image 8
> 
> ![](_page_0_Figure_3.jpeg)
> 
> FIG. 8G
![image.png](GPU%20Large%20Data%20Access/image%2020.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 8H
![image.png](GPU%20Large%20Data%20Access/image%2021.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 8I
![image.png](GPU%20Large%20Data%20Access/image%2022.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 8J
![image.png](GPU%20Large%20Data%20Access/image%2023.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 8K
![image.png](GPU%20Large%20Data%20Access/image%2024.png)

## Fig9

layout swizzle：避免Bank冲突

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Large%20Data%20Access/image%2025.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 9B
![image.png](GPU%20Large%20Data%20Access/image%2026.png)

> **[图片提取文字 (image.png)]:**
> | W | Н    | C=0-63 |     |     |     |       |     |      |       |
> |---|------|--------|-----|-----|-----|-------|-----|------|-------|
> | 0 | 0    | 0      | 1   | 2   | 3   | 4     | 5   | - 6  | 7     |
> | 1 |      | 8      | 9   | 10  | 11  | 12    | 13  | 14   | 15    |
> | 2 |      | 16     | 17  | 18  | 19  | 20    | 21  | 22   | 23    |
> | 3 |      | 24     | 25  | 26  | 27  | 28    | 29  | 30   | 31    |
> | 4 |      | 32     | 33  | 34  | 35  | 36    | 37  | 38   | 39    |
> | 5 |      | 40     | 41  | 42  | 43  | 44    | 45  | 46   | 47    |
> | 6 |      | 48     | 49  | 50  | 51  | 52    | 53  | 54   | 55    |
> | 7 |      | 56     | 57  | 58  | 59  | 60    | 61  | 62   | 63    |
> | 8 |      | 64     | 65  | 66  | 67  | 68    | 69  | 70   | 71    |
> | 9 |      | 72     | 73  | 74  | 75  | 76    | 77  | 78   | 79    |
> | 0 | , ch | 80     | 81  | 82  | 83  | 84    | 85  | . 86 | 87    |
> | 1 |      | 88     | 89  | 90  | 91  | 92    | 93  | 94   | 95    |
> | 2 |      | 96     | 97  | 98  | 99  | 100   | 101 | 102  | 103   |
> | 3 |      | 104    | 105 | 106 | 107 | 108   | 109 | 110  | 111   |
> | 4 |      | 112    | 113 | 114 | 115 | . 116 | 117 | 118  | /119  |
> | 5 |      | 120    | 121 | 122 | 123 | 124   | 125 | 126  | 127   |
> | 6 |      | 128    | 129 | 130 | 131 | 132   | 133 | 134  | ×135× |
> | 7 |      | 136    | 137 | 138 | 139 | 140   | 141 | 142  | 143   |
> | 8 |      | 144    | 145 | 146 | 147 | 148   | 149 | 150  | /151  |
> | 9 |      | 152    | 153 | 154 | 155 | 156   | 157 | 158  | / 159 |
> 
> FIG. 9C
![image.png](GPU%20Large%20Data%20Access/image%2027.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 9D
![image.png](GPU%20Large%20Data%20Access/image%2028.png)

## Fig10、11、12、13

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 10
![image.png](GPU%20Large%20Data%20Access/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 11A
> Example General Processing Cluster
![image.png](GPU%20Large%20Data%20Access/image%2030.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 11B
![image.png](GPU%20Large%20Data%20Access/image%2031.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> To/from MMU 1290
> 
> FIG. 12
![image.png](GPU%20Large%20Data%20Access/image%2032.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 13A
![image.png](GPU%20Large%20Data%20Access/image%2033.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 13B
![image.png](GPU%20Large%20Data%20Access/image%2034.png)