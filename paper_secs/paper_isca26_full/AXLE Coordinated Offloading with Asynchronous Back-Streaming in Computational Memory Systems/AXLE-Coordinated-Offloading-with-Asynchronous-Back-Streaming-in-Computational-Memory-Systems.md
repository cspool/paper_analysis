# AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems

Suyeon Lee
School of Computer Science
Georgia Institute of Technology
Atlanta, USA
sylee0506@gatech.edu

Kangkyu Park
Memory System Research
SK hynix Inc.
Icheon, South Korea
kangkyu.park@sk.com

Kwangsik Shin

Memory System Research

SK hynix Inc.

Icheon, South Korea
kwangsik.shin@sk.com

Ada Gavrilovska

School of Computer Science

Georgia Institute of Technology

Atlanta, USA

ada@cc.gatech.edu

Abstract—CXL-based Computational Memory (CCM) enables near-memory processing within expanded remote memory, offering opportunities to address data movement costs in disaggregated memory systems and to accelerate overall performance. However, existing offloading mechanisms do not fully leverage the trade-offs of different offload models based on different CXL protocols. This work first examines these tradeoffs and their impact on end-to-end performance and system efficiency for workloads with diverse data and computation characteristics. We propose Asynchronous Back-Streaming, a new offloading protocol that coordinates CXL.io and CXL.mem to enable result back-streaming and asynchronous pipelining across CCM and host tasks. We further design AXLE, a system that realizes this protocol with lightweight host-CCM interaction. Overall, AXLE reduces end-to-end runtime by up to 50.14%, reduces CCM and host idle times by an average of  $14.53 \times$  and  $3.93 \times$ , respectively, and achieves up to  $6 \times$  reduction in host core stall time.

Index Terms—Computational Memory, CXL, Operation Offloading

# AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems

Suyeon Lee
School of Computer Science
Georgia Institute of Technology
Atlanta, USA
sylee0506@gatech.edu

Kangkyu Park
Memory System Research
SK hynix Inc.
Icheon, South Korea
kangkyu.park@sk.com

Kwangsik Shin

Memory System Research

SK hynix Inc.

Icheon, South Korea
kwangsik.shin@sk.com

Ada Gavrilovska

School of Computer Science

Georgia Institute of Technology

Atlanta, USA

ada@cc.gatech.edu

Abstract—CXL-based Computational Memory (CCM) enables near-memory processing within expanded remote memory, offering opportunities to address data movement costs in disaggregated memory systems and to accelerate overall performance. However, existing offloading mechanisms do not fully leverage the trade-offs of different offload models based on different CXL protocols. This work first examines these tradeoffs and their impact on end-to-end performance and system efficiency for workloads with diverse data and computation characteristics. We propose Asynchronous Back-Streaming, a new offloading protocol that coordinates CXL.io and CXL.mem to enable result back-streaming and asynchronous pipelining across CCM and host tasks. We further design AXLE, a system that realizes this protocol with lightweight host-CCM interaction. Overall, AXLE reduces end-to-end runtime by up to 50.14%, reduces CCM and host idle times by an average of  $14.53 \times$  and  $3.93 \times$ , respectively, and achieves up to  $6 \times$  reduction in host core stall time.

Index Terms—Computational Memory, CXL, Operation Offloading

