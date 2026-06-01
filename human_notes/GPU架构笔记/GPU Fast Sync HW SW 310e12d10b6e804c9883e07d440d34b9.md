# GPU Fast Sync HW / SW

ref：FAST DATA SYNCHRONIZATION INPROCESSORS AND MEMORY

ref：HARDWARE ACCELERATED SYNCHRONIZATION WITH ASYNCHRONOUS TRANSACTION SUPPORT

## Fig1、2

GPU架构

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 1A
![image.png](GPU%20Fast%20Sync%20HW%20SW/image.png)

> **[图片提取文字 (image.png)]:**
> ## Weak scaling DL. Output Activations FIG. 1B
> 
> ## Strong scaling
> 
> ![](_page_0_Picture_2.jpeg)
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%201.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 2
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%202.png)

## Fig3、4、5、6

CGA内的SM之间直接通过shared-mem通信，但容易成为性能瓶颈。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 3A
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%203.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%204.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 3C
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%205.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 3D
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%206.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 4
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%207.png)

不同SM上不同线程的同步，生产者SM直接将数据写到消费者SM的Mem中（remote store），数据同步通过wait和arrive进行等待和唤醒，或者操作之间的memory order设置。

但remote wait的实现开销很大。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%208.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 5C
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%209.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2010.png)

远程存储STS+唤醒Arrive的组合指令，基于c-barrier和p-barrier同步的硬件实现开销可接受。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 6A
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 6B
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2012.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> N =tile id
> 
> p-barrier c-barrier
> 
> wait
> 
> arr.
> 
> FIG. 6C
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2013.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 6D
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2014.png)

## Fig7、8、9

STS+Arrv指令的datapath，Barrier的硬件实现

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 8
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2016.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ## FIG. 7B
> 
> | 63                  | 56       | 55                                   | 32 | 31 2                  | 24 23 |                                   | 0 |
> |---------------------|----------|--------------------------------------|----|-----------------------|-------|-----------------------------------|---|
> | Barrie<br>CTA_ID in | r<br>CGA | Barrier Address in target CTA's SMEM |    | Data<br>CTA_ID in CGA |       | Data Address in target CTA's SMEM |   |
> 
> ## FIG. 7C
> 
> XBAR to SM Transaction for Arrive w/ Every Transaction
> 
> | Packet 0 | CGA info; Data SMEM Adr  | 32 B Data |  |  |  |
> |----------|--------------------------|-----------|--|--|--|
> | Packet 1 | ZFill info; Bar SMEM Adr | 32 B Data |  |  |  |
> | Packet 2 | unused                   | 32 B Data |  |  |  |
> | Packet 3 | unused                   | 32 B Data |  |  |  |
> 
> FIG. 7D
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2017.png)

Barrier中的fence指令

SM之间的数据交换模型/情景（*）：基于全局Mem的数据交换、基于shared-mem的SM2SM、L-Cache中介。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 9A
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2018.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> FIG. 9C
> 
> ![](_page_0_Figure_2.jpeg)
> 
> FIG. 9D FIG. 9E FIG. 9F
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2019.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 9B
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2020.png)

L-Cache中介的数据交换的SOL延迟

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 9G
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2021.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 9H
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2022.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 91
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2023.png)

## Fig10、11、12、13

GPU架构和扩展架构

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 10
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2024.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 11A
> Example General Processing Cluster
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2025.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 11B
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2026.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 13A
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2027.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> To/from MMU 1290
> 
> FIG. 12
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2028.png)

## Fig4、5、6

Barrier transaction

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 4
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%207.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2030.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2031.png)

## Fig7、8

Barrier transaction实现

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 7
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2032.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 8
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2033.png)

## Fig9

SM通信的调用栈

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 9A
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2034.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> 950
> 
> ## FIG. 9C
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2035.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 98
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%2036.png)