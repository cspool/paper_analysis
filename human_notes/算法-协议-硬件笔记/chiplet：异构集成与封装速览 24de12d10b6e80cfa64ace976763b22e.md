# chiplet：异构集成与封装速览

ref：[https://zhuanlan.zhihu.com/p/361000848](https://zhuanlan.zhihu.com/p/361000848)

ref：[https://zhuanlan.zhihu.com/p/621786825](https://zhuanlan.zhihu.com/p/621786825)

ref：[https://www.latitudeda.com/document/688](https://www.latitudeda.com/document/688)

ref：[https://zhuanlan.zhihu.com/p/603355527](https://zhuanlan.zhihu.com/p/603355527) 

ref：[https://blog.csdn.net/sinat_32960911/article/details/145101182](https://blog.csdn.net/sinat_32960911/article/details/145101182)

ref：[https://mp.weixin.qq.com/s/0XMWlXGqkeBzYFpty_hpFw](https://mp.weixin.qq.com/s/0XMWlXGqkeBzYFpty_hpFw)

# 封装和集成

硬件系统，是**集成integration**不同功能部件，让其协作完成设计功能的硬件；

根据集成度不同，可分为芯片（裸片die）上集成SoC、多die集成chiplet、多封装堆叠SiP/MCM和PCB板级集成SoP，系统集成度依次降低；

**封装package包装单个die，或者堆叠、连接多个die/封装**，保留连接外部的接口；

成熟的系统级硬件实现是**SoC和SiP**，工艺、面积限制导致**SoC的集成度难以提高**，并且SoC所有组件统一工艺使得良品率低而chiplet组合使用良品率高的组件来降低成本，因此**系统级硬件朝chiplet方向发展；**

**chiplet的难点是多die的集成，**在多个die和基板之间建立垂直连接，实现的技术栈有2.5D封装、3D封装，3D封装技术分为前端和后端；

**前端3D侧重集成，也称3D集成**，是bumpless的die间直接连接，垂直相连的die视为整体系统；

后端3D侧重封装，是基于bump的die间垂直连接，die原生认为是独立模块并设置IO；

封装、集成技术有2D封装、2.5D封装、3D封装等，2D通过substate进行die间互联，2.5D通过设置tsv的interposer进行die间互联，3D通过bump或die中tsv进行互联；

技术特点：[https://zhuanlan.zhihu.com/p/2220249892](https://zhuanlan.zhihu.com/p/2220249892)；

# chiplet的3D封装

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ## PCB to MCM/SiP Benefits
> 
> Smaller footprint
> 
> PCB simplification
> 
> Higher bandwidth
> 
> Lower power
> 
> ## SoC to HI Benefits
> 
> Reduced NRE costs
> 
> Shorter time to market
> 
> Larger than reticle size designs
> 
> More flexible IP use-model
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image.png)

系统级封装 (即System-in-Package ，SiP) 有两个不同的方向。一是把 PCB 上的器件转移到**多芯片组件（MCM）**；二是如同前几年制造大型系统级芯片（即System-on-Chip， SoC）一样进行**集成，但是转换制程利用先进封装来封装裸片（chiplet）**。

> **[图片提取文字 (image.png)]:**
> ## 以下是一些使<mark>晶粒(Chiplet</mark>)解决方案具有吸引力的重要因素:
> 
> 需要"全部统一在单一"制程节点上
> 
> - 由于制造裸片尺寸小,所以良率会更高使用现成的晶粒(Chiplet),可缩短 IC 的设计周期,并降低集成的复杂性
> - 通过购买良品裸片(即known-good-die , KGD), **可普遍降低生产成本**
> - 在许多设计中使用同种晶粒 (Chiplet) 时,将具有如同采用批量生产的相同成本优势
> 
> • **在为器件挑选最佳工艺节点方面具有很大的灵活性**;特别是 SerDes I/O 和模拟核,不再
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%201.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%202.png)

以 IC 为重心的先进封装改变了设计流程。上图中，20世纪90年代设计采用的是类似 PCB 的设计流程；而**如今已采用类似 IC 的设计流程**。**把多种不同的技术集成到一起，即异构集成HI，结合了多年以来使用的各种制程技术。**特别是**先进封装和先进集成方法，例如晶圆堆叠（Wafer-on-Wafer）和无凸块集成（Bumpless）**。

> **[图片提取文字 (image.png)]:**
> ## System-Level Design and **Analysis Tools**
> 
> IC Design and **Verification Tools** 
> 
> 3D Packaging
> 
> ![](_page_0_Picture_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
> 
> ![](_page_0_Picture_6.jpeg)
> 
> ![](_page_0_Picture_7.jpeg)
> 
> ![](_page_0_Picture_8.jpeg)
> 
> ![](_page_0_Picture_9.jpeg)
> 
> ![](_page_0_Picture_10.jpeg)
> 
> ![](_page_0_Picture_11.jpeg)
> 
> ![](_page_0_Picture_12.jpeg)
> 
> ![](_page_0_Picture_13.jpeg)
> 
> ![](_page_0_Picture_14.jpeg)
> 
> ![](_page_0_Picture_15.jpeg)
> 
> ![](_page_0_Picture_16.jpeg)
> 
> ![](_page_0_Picture_17.jpeg)
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%203.png)

**我们可以将基于封装的 3D 视为“后端 3D”（3D封装），把先进集成方式视为“前端 3D”（3D集成）。**

**后端 3D是微型凸块互连（micro-bumped）加上每个裸片都有单独的时序签核和 I/O 缓冲器。**这种方式中，多个裸片之间通常没有采用并行设计。多年来，这一直是用于存储器和 CMOS 图像传感器的常见方法。

**对于前端 3D，裸片通常是直接键合的制程工艺（铜对铜，或采用类似方法）。**裸片之间没有 I/O 缓冲器，这意味着并行设计和分析必不可少，需要时序驱动的布线和静态时序签核（对于数字设计而言）。所以设计将倾向于朝Z 轴上布局，多个裸片会堆叠在一起；这意味着随着设计的推进，一个特定的区域可能被分配给超过一个的裸片。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%204.png)

这是封装领域的下一个重要转变，也是向真正3D-IC 设计迈出的一大步，即**将众多不同的裸片堆叠在一起，这能大大缩短信号所需的传输距离。**当然，由此产生的**散热问题**也需要加以分析和管理，裸片上方的另一个裸片可能会阻绝散热，这取决于众多的设计细节。

# 不同封装概述

**封装是保护die/堆叠封装并暴露接口，封装是集成硬件系统的实现方式；**

芯片从设计到生产再到消费者手中是个极其复杂的过程，设计公司做完逻辑和物理设计，将最终设计结果交给芯片代工厂。代工厂经过无数复杂的流程，最终会在一块大的晶圆上做出许许多多的小芯片。而这一个个的小芯片，则被称为“**die**”。

die非常非常脆弱，因此不能直接使用，需要再给它加上一层保护壳，而这个过程，就叫做“封装”。简单点说，**封装技术需要将 die 固定在基板（substrate）上，然后将die上的引脚连接到芯片外壳的引脚上。**

**硬件系统可分为芯片（die）上集成（SoC）、封装内集成（SiP）和PCB板级集成（SoP）三种实现；**

> **[图片提取文字 (image.png)]:**
> 电子集成技术分为 三个层次:
> 
> - 芯片上的集成;
> - 封装内的集成;
> - PCB板级集成,其代表技术分别为SoC, SiP和PCB(也可以称为SoP或者SoB)。
> 
> 芯片上的集成主要以**2D**为主,晶体管以平铺的形式集成于晶圆平面;同样,PCB上的集成也是以2D为主,电子元器件平铺安装在PCB表面,因此,二者都属于2D集成。而针对于封装内的集成,情况就要复杂的多。
> 
> ## 电子集成技术分类的两个重要判据:
> 
> - 1. 物理结构;
> - 2. **电气连接(** 电气互连)。
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%205.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%206.png)

**单die的封装方法**

> **[图片提取文字 (image.png)]:**
> ## 线键合 (wire-bonding) 封装
> 
> 最基础的封装工艺即为:**引线键合**(wire-bonding)封装,其整体上十分简单,就是把die正面朝上固定到基板之上,再用导线,将die的引脚和基板连接(称之为'**键合**'),最后把整个芯片封装起来,密封用的材料有塑料,陶瓷等。这种封装技术的优点是生产工艺相对简单,成本较低;**缺点是封装完的芯片尺寸比 die 的尺寸大许**多,且芯片管脚数受限。
> 
> ![](_page_0_Picture_2.jpeg)
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%207.png)

> **[图片提取文字 (image.png)]:**
> ## **FOWLP**
> 
> 之后,随着技术的进步,又出现了"**倒装**",即将die的正面朝下,提前做好焊点的技术,倒装的应用使得封装尺寸和芯片接近,并且有更多的引脚,但是随着芯片功能越来越多,I/O数量急剧增加,传统的封装已经难以满足要求。后来据此还衍生出了 Fan-Out WLP (Wafer Level Packages) ,也叫FOWLP技术,但是文章篇幅有限,有兴趣的读者可以自行了解。
> 
> ![](_page_0_Picture_2.jpeg)
> 
> 上文中所言都**是单独die的封装**,一颗完整的现代芯片,单个die是远远不够的,需要将多个die封装在一起,而这之中的封装方式便是 2D, 2.5D, 3D封装。
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%208.png)

**多die的封装方法**

> **[图片提取文字 (image.png)]:**
> ## 2D封装
> 
> **2D 封装**是指在 **基板(substrate)** 的表面水平安装所有芯片和无源器件的集成方式。以基板 (Substrate) 上表面的**左下角**为原点,基板上表面所处的平面为XY平面,基板法线为Z轴,创建坐标系。
> 
> ![](_page_0_Picture_2.jpeg)
> 
> - **物理结构**: 所有芯片和无源器件均安装在基板平面,芯片和无源器件和 XY 平面直接接触,**基板上的布线和过孔均位于 XY 平面下** 方;
> - 电气连接:均需要通过基板 (除了极少数通过键合线直接连接的键合点)
> 
> 台积电在**2017年**开发的**InFO技术**。InFO技术与大多数封装厂的Fan-out类似,可以理解为多个芯片Fan-out工艺的集成,**主要区别在于去掉了silicon interposer**,**使用一些RDL层进行串连**(2016年推出的iPhone7中的**A10处理器**,采用台积电16nm FinFET工艺以及InFO技术)。
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%209.png)

> **[图片提取文字 (image.png)]:**
> ## 硅中介层无TSV的2.5D
> 
> 硅中介层无TSV的2.5D集成的结构一般如下图所示,有一颗面积较大的裸芯片直接安装在基板上,该芯片和基板的连接可以采用**Bond Wire**或者**Flip Chip**两种方式,大芯片上方由于面积较大,可以安装多个较小的裸芯片,但小芯片无法直接连接到基板,所以需要插入一块中介层(Interposer),在中介层上方安装多个裸芯片,中介层上有RDL布线,可将芯片的信号引出到中介层的边沿,然后通过Bond Wire连接到基板。这类中介层通常不需要TSV,只需要通过Interposer上表面的布线进行电气互连,Interposer采用Bond Wire和封装基板连接。
> 
> ![](_page_0_Picture_2.jpeg)
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ## 硅通孔(TSV)
> 
> 硅通孔(Through Silicon Vias,简称TSV)是一种**在硅晶圆(而不是基板或PCB上)上制作垂直贯通的微小通孔,并在通孔中填充导电材料,实现芯片内部不同层面之间的电气连接的技术**。这种技术能够显著提高芯片内部的互连密度,降低信号传输延迟,提高系统的整体性能。TSV技术广泛应用于存储器、处理器、图像传感器等高性能芯片中,尤其是在3D IC封装中具有重要应用。
> 
> ## TSV 硅通孔 TSV: Through Silicon Via
> 
> - 定义为链接硅晶圆两面并与硅衬底和其他通孔绝缘的电互连结构。
> - 使用方法是: 硅连接板和直接使用TSV
> 
> ![](_page_0_Figure_5.jpeg)
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ## Pros and Cons of 3D IC/2.5D IC
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Courtesy of TSMC Reference Flow
> 
> | Pros                                                                                                                         | Cons                                                                                                                                                                               |
> |------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
> | Small form factor     Short channel length & propagation delay     Lower power, higher bandwidth     High system performance | <ul> <li>Small heat sink area =&gt; Heat accumulation</li> <li>Design implementation is hard</li> <li>High complexity</li> <li>Difficult to analyze, verify and signoff</li> </ul> |
> 
> Comprehensive chip, package and system co-design and co-analysis is necessary
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%2012.png)

> **[图片提取文字 (image.png)]:**
> 2.5D 和3D 最本质的区别是: 2.5D有中介层 interposer, 3D没有 interposer层面。
> 
> ![](_page_0_Figure_1.jpeg)
> 
> - 物理结构: 芯片堆叠或并排放置在具有TSV的中介层 (interposer) 上,中介层提供芯片之间的连接性。
> - 电气连接:通过中介层上的微型凸点 (micro-bumps) 和TSV实现电气互连。
> - •特点:集成度较高,可以提供更高的I/O密度和更低的传输延迟,但相比3D封装,其**垂直堆叠的芯片数量较少**。
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%2013.png)

> **[图片提取文字 (image.png)]:**
> ## 3D封装
> 
> 实现在3D封装的关键技术就是**TSV硅通孔技术**。简单来讲,TSV技术通过在芯片与芯片之间、晶圆和晶圆之间制作垂直导通,通过导电物质的填充实现硅通孔的垂直电气互联,它是目前唯一能实现垂直电互联的技术。
> 
> 这种技术看上去十分完美,但是难度太高,成本太大。试想一下,在又薄又脆弱的玻璃片上打很多通孔,再把这些经过处理之后更加脆弱的芯片垒成"摩天大楼",听着就十分困难。因此,TSV技术在1958年被威廉·肖特基(William Shockley)第一次申请专利之后,直到40多年后的21世纪才逐渐走向商用:
> 
> - 2000 年,日本分别率先研发出第一款三层堆叠的图像传感器和三层堆叠的存储器件;
> - 2005 年, 10 层堆叠的存储芯片被研制出来;
> - 2007 年集成 TSV 的 CIS 芯片由 Toshiba 公司量产商用,同年 ST Microelectronics 和 Toshiba 一起推出 8 层堆叠的 NAND 闪存芯片;
> - 2013 年第一款 HBM 存储芯片由韩国 Hynix 推出;
> - 2015 年, 第一款集成 HBM 的 GPU 由 AMD 推出。
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ## 3D集成和2.5D集成的主要区别在于:
> 
> - 2.5D 集成是在中介层Interposer上进行布线和打孔;
> - 3D 集成是直接在芯片上打孔 (TSV) 和布线 (RDL), 电气连接上下层芯片。
> 
> ## 2.5D和3D IC设计
> 
> ![](_page_0_Figure_4.jpeg)
> 
> **物理结构**:所有芯片和无源器件均位于XY平面上方,芯片堆叠在一起,在XY平面的上方有穿过芯片的TSV,在XY平面的下方有基板的布 线和过孔。
> 
> 电气连接:通过TSV和RDL将芯片直接电气连接
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%2015.png)

> **[图片提取文字 (image.png)]:**
> **3D 集成大多数应用在同类芯片堆叠中**,多个相同的芯片垂直堆叠在一起,通过穿过芯片堆叠的TSV互连,如下图所示。同类芯片集成大多应用在存储器集成中,例如DRAM Stack,FLASH Stack等。
> 
> ![](_page_0_Picture_1.jpeg)
> 
> 不同类芯片的3D集成中,一般是将两种不同的芯片垂直堆叠,并通过TSV电气连接在一起,并和下方的基板互连,有时候需要在芯片表面制作RDL来连接上下层的TSV。
> 
> ![](_page_0_Picture_3.jpeg)
![image.png](chiplet%EF%BC%9A%E5%BC%82%E6%9E%84%E9%9B%86%E6%88%90%E4%B8%8E%E5%B0%81%E8%A3%85%E9%80%9F%E8%A7%88/image%2016.png)