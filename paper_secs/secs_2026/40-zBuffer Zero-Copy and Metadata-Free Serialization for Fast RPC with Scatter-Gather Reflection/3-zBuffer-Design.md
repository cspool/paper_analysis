# 3 zBuffer Design

#### 3.1 Programming Model

The scatter-gather structure (sg\_array) is the core abstraction of scatter-gather based serialization. As shown in Listing [1,](#page-3-0) sg\_array maintains an array of sg and its begin, end, and capacity. The structure of sg consists of a pointer (sg\_base) that points to a memory buffer of its virtual address and a record of buffer length (sg\_len). An sg is used to represent a contiguous memory buffer, and an sg\_array is used to represent multiple non-contiguous memory buffers. This approach is conceptually similar to the writev [\[14\]](#page-10-16) system call in Linux, which uses an iovec data structure to transmit data. However, in Linux, the kernel still copies the iovec into a contiguous buffer before transmission.

To use zBuffer, a developer defines a data structure schema using the C++ language such as the Msg struct shown in Listing [2.](#page-3-1) We provide separate implementations for common variable-length data types, such as string and array, using a pointer and a length to represent and thus adapt to scatter-gather. The macro PROCESS\_FIELDS specifies the order of fields to be serialized, except that fields requiring alignment are always processed first (see [§3.3\)](#page-5-0). Next, the developer builds and fills the zBuffer object in their code, and uses the Serializer to serialize the object to get the sg\_array, which is then passed to the network stack for

```
template<typename T>
 void process_field(T& x) {
      x.serialize_fields(*d());
 }
 #define PROCESS_FIELDS(...) \
    template<typename AR> \
    void process_fields(AR& ar) { \
        return reduce(ar, __VA_ARGS__); \
 template<typename AR, typename T, typename...Ts>
 void reduce(AR& ar, T& x, Ts&...xs) {
     ar.process_field(x);
     reduce(ar, xs...);
 }
1 Using a variadic macro to accept a list of fields
2 Using a variadic template to recursively process each field
3 Type specialization processing
```

Figure 5. Core logic for zBuffer to realize static reflection.

transmission. After receiving the packet, the developer uses the Deserializer to deserialize the receive buffer back into a pointer-based data structure. Our prototype supports serialization of base integer types, strings, bytes, nested objects, and lists of strings, bytes or nested objects. We use different data types to identify fields that need alignment and those that do not (e.g., aligned\_message and message).

#### 3.2 Scatter-Gather Reflection

- 3.2.1 Static Reflection. zBuffer leverages C++ features such as template metaprogramming and macros to realize compile-time static reflection. The processing code for the fields is generated at compile time, as shown in Fig. [5:](#page-3-2)
- ❶ zBuffer uses the variadic macro VA\_ARGS to receive a list of fields, thereby registering the fields and obtaining type information at compile time. This process is implemented through the PROCESS\_FIELDS macro, which expands into a templated method process\_fields and passes the list of fields to the reduce function. This approach allows the user to register the fields that need to be serialized using the concise syntax PROCESS\_FIELDS without having to write the reflection code manually.
- ❷ Subsequently, variadic templates and recursion are used to expand and process all fields, where x represents the current field and xs represents the remaining fields. This part is implemented by the template function reduce, which recursively calls ar.process\_field(x) to process the current field and then continues with the remaining fields until all fields have been processed. This recursive template expansion occurs entirely at compile time, generating code that is equivalent to directly invoking each field's processing function in sequence.
- ❸ Finally, type specialization is performed. For different field types, the compiler selects the corresponding template

specialization to generate the processing code. zBuffer provides multiple overloads of process\_field such as string and array. This type specialization is entirely determined at compile time, avoiding the runtime overhead associated with type checks and dispatch.

zBuffer does most work at compile time, including type checking, field traversal, and processing logic specialization highly optimized by the compiler. zBuffer realizes static reflection through compile-time techniques without performance penalty suffered by traditional runtime reflection.

**3.2.2 Serializer.** The responsibility of the Serializer is to construct the appropriate  $sg\_array$  to represent the input message, which is eventually passed to the network stack. Leveraging the  $sg\_array$  presented in Listing 1 enables us to capture the entire message without incurring the overhead of copying multiple non-contiguous memory buffers into a single contiguous buffer.

Algorithm 1 shows how zBuffer serializes the message object. During serialization, the Serializer first initializes an empty sg\_array (line 2) and then performs a two-stage field processing on the input object to satisfy the memory alignment requirements. It calls FilterAlignedFields (with flag set to true) to filter out all aligned fields (line 3), and then processes these fields by Message::process\_field (line 4), which internally calls the recursive Message::reduce function (line 13). Then the process\_field function (lines 19-31) is called one by one for each field in order; for each field whose length is not 0, an sg element is created and added to the sg array. The Serializer then calls the same filter function again (with flag set to false) to filter out the unaligned fields and processes them similarly (lines 5-6). After processing all fields, the Serializer creates an sg for the object and finally pushes the sg to the sg array. For nested data, such as trees, the Serializer will perform the same process used for non-nested data at each level. The serialization process does not involve memory copy, just constructing the sg\_array which is eventually passed to the network stack. The NIC coalesces the non-contiguous buffer pointed to by each sg.

**3.2.3 Deserializer.** Deserialization requires that the received payload be correctly turned back into the original data structure. However, objects are sent containing pointers that are only valid in the sender's address space and not in the receiver's. zBuffer's deserialization algorithm overwrites these pointers to point to the correct memory address.

Algorithm 2 describes the deserialization process. The first step is to obtain a pointer to the original message object so that the fields can be correctly recovered based on the information recorded in the message object. Since we are serializing in the specified order, the message object is located at the end of the receive buffer, and we simply create a message pointer to the last message-sized memory at the end of the buffer. The size of this buffer is calculated using

### **Algorithm 1:** Serialize Object into the *sg\_array*

```
procedure serialize(obj)
       sg\_array \leftarrow empty
       // Process aligned fields
       aligned \leftarrow FilterAlignedFields (this, true)
 3
       Message::process_field(obj, aligned)
       // Process non-aligned fields
       non aligned ← FilterAlignedFields (this, false)
 5
       Message::process_field(obj, non_aligned)
       Create a sg_{obj} with sg.base \leftarrow obj.ptr, sg.len \leftarrow obj.len
       sg_array.push_back(sg<sub>obi</sub>)
9 end
procedure Message::process_field(obj, archive)
       // Expand all fields
       Message::reduce(archive, obj.fields...)
12 end
13 procedure Message::reduce(archive, field,
    remain fields)
       // Process current field, call line 19
       archive.process_field(field)
       // Recursively process remaining
15
       if remain_fields \neq \emptyset then
           Message::reduce(archive, remain fields)
16
17
       end
18 end
19 procedure
    FilterAlignedFields::process_field(field)
       if is_aligned_field(field) and flag = true then
20
           Serializer::process_field(field)
21
       else if not is_aligned_field(field) and flag = false then
22
           Serializer::process_field(field)
23
24
       end
25 end
26 procedure Serializer::process_field(field)
       if field.len \neq 0 then
           Create an sg with
            sg.base \leftarrow field.ptr, sg.len \leftarrow field.len
           sg_array.push_back(sg)
       end
31 end
```

the *sizeof* operator [13] for the type of object (line 4). Next, starting at the start of the receive buffer (line 3), we process each field in turn in the same order as it was serialized at. We fix the pointer of each field to the appropriate memory address based on the length of the field stored in the object and the current offset of the receive buffer (line 13). In particular, for integer fields, which are already in the object's memory, we do not need to handle them specifically and can access them directly, for example by using obj.id. The in-place deserialization does not require memory copy.

#### Algorithm 2: Deserialize Object from Buffer

```
procedure deserialize(buffer)
      buffer, len /* receive buffer and buffer length
2
      offset \leftarrow 0
                     /* current offset in the receive
       buffer */
      obj \leftarrow (obj)(buffer + len - sizeof(obj))
      aligned ← FilterAlignedFields (this, true)
5
      Message::process_field(obj, aligned)
      non aligned ← FilterAlignedFields (this, false)
      Message::process_field(obj, non_aligned)
      return obj
10 end
  /* Some procedures are omitted for brevity, as
      they are similar to Algorithm 1
procedure Deserializer::process_field(field)
      if field.len \neq 0 then
12
          field.ptr ← buffer + offset/* fix field pointer
          offset ← offset + field.len /* update offset */
14
      end
16 end
```

<span id="page-5-2"></span>![](_page_5_Figure_4.jpeg)

Figure 6. Example of mem-aligned.

#### <span id="page-5-0"></span>3.3 Optimization for Memory Alignment

Memory alignment is important in many scenarios like direct I/O [54, 66], SIMD instruction [12]. However, the existing serialization library does not optimize for this requirement. As depicted in Fig. 6, after serialization, transmission, and deserialization, although the address of the received buffer is block-size aligned (e.g., using posix\_memalign [8] for aligned memory allocation), the size of the preceding fields is not. As a result, the buffer offset for the field requiring alignment does not meet block-size alignment. Therefore, the field must be copied to aligned memory before being written to disk using direct I/O, which introduces additional memory copy overhead.

We optimize the serialization and deserialization processes to address the potential overhead caused by memory misalignment. We distinguish between buffers that require memory alignment and those that do not. Based on the serialization order specified by the application, the Serializer prioritizes pushing fields that require memory alignment into the

```
void init_layout(sg_array *sgs);
void enqueue_request(sg_array *msg, ...);
```

<span id="page-5-5"></span>**Listing 3.** zRPC Interface for integrating zBuffer.

![](_page_5_Picture_11.jpeg)

Figure 7. Message Layout.

<span id="page-5-6"></span>![](_page_5_Picture_13.jpeg)

Figure 8. Dual ring buffer.

sg\_array. During deserialization, after extracting the message itself, we restore the pointers in the same order as the serialization, thus preserving memory alignment.

#### 4 zRPC: zBuffer-Based Fast RPC

#### 4.1 zRPC Interface

Listing 3 presents the essential API function of zRPC for integrating zBuffer. Once serialization is complete, the resulting  $sg\_array$  is passed to the RPC system, which subsequently invokes the init\_layout function to generate the message layout (§4.2). Following this, the enqueue\_request function is called to send the request, during which the  $sg\_array$  is passed to the NIC's DMA engine to initiate DMAs, and the DMA engine then coalesces the memory buffers specified in the  $sg\_array$  for network transmission.

#### <span id="page-5-4"></span>4.2 Message Organization

zRPC realizes network communication through packet transmission. zRPC employs the *sg\_array* structure to represent a complete message. This design is fully compatible with the scatter-gather feature of networking devices and RDMA. The content recorded by *sg\_array* consists of two parts: headers and data. A complete *sg\_array* has at least two *sg*, with one recording the location of the packet header and the other recording the location of the message. When the message size exceeds the maximum transmission unit (MTU), multiple packets need to be sent, one *sg\_array* can still represent the complete message using multiple *sg* to represent the memory buffers of the headers and the data in the message.

However, determining the position of the relevant sg for each packet introduces considerable overhead, especially for messages that require a large number of packets. To address this challenge, we have introduced an indexing mechanism that efficiently locates the appropriate sg within the  $sg\_array$  corresponding to each packet. We use an index array where each entry marks a sub-packet boundary by recording the

containing sg and its intra-sg offset. To send the  $j_{th}$  subpacket, we read entries  $I_j$  and  $I_{j+1}$  to get its start and end, which directly identifies the spanned sgs and their segment offsets. This approach significantly reduces the overhead associated with packet transmission, thus enabling efficient message delivery even for large and complex messages.

Consider the example illustrated in Fig. 7, in which a message is transmitted using four packets. For  $Packet_1$ , comprising  $SGs_1$  and  $Header_1$ , the beginning and end positions of  $SGs_1$  can be easily determined using  $Index_1$  and  $Index_2$ , respectively. Similarly, the position of  $Header_H$  can be quickly located through  $Index_H$ .

#### 4.3 Header-Data Separation

Achieving zero-copy transmission remains challenging, even with kernel-bypass I/O techniques like RDMA, which avoid data copying between user and kernel space. Notably, state-of-the-art RPC frameworks [37] still rely on traditional receive ring buffers that store packet headers and data together. For example, if a field spans two packets, its data is split by a header, so the data need to be copied into a contiguous memory buffer, which incurs additional overhead.

To address this challenge, zRPC design a dual-ring buffer to coordinate the separate transmissions of packet headers and data. As shown in Fig. 8, we allocate two ring buffers, one for packet headers and another for data, which share a common index for lookup and update. Upon packet arrival, the data header and data are stored separately in their respective buffers at the locations indicated by the pointers, so that the data of fields spanning packets are stored continuously in the same ring buffer, thus avoiding memory copying. We use credits per connection for packet-level flow control: limiting the number of packets a client sends in a connection before receiving a reply, thereby ensuring the ring buffer can accommodate all messages.

#### 4.4 Implementation

The implementation of zRPC includes the serialization library (zBuffer) and an eRPC-based networking stack. eRPC [37] leverages the fact that switch buffer capacity far exceeds datacenter bandwidth—delay product (BDP), and realizes zero-copy network transmission in the presence of retransmissions, node failures, and rate limiting. It simply uses a contiguous memory buffer to represent a message, and does not realize serialization. We enhance eRPC to support zRPC's API, message organization, and header-data separation, and integrate it with zBuffer to realize zRPC.

Transport Engines. zRPC uses RDMA send/recv for plain packet I/O instead of RDMA write/read to send messages. This is because packet I/O has better scalability [38]. Our implementation uses the RoCE (RDMA over Converged Ethernet) network protocol, and is implemented based on OFED libibverbs 5.4. To create an RPC service, the developer needs to register request handler functions with unique request

types in RPC servers and clients use these request types when issuing RPCs.

Scatter-Gather DMA. To enable NIC scatter-gather work, the network stack receives sg\_array as input, then takes the sg\_array to fill struct ibv\_sge\* sg\_list, which is used in the RDMA network stack API. Each ibv\_sge represents a NIC-registered buffer, and sg\_list is an array of ibv\_sge. Subsequently, the NIC directly gathers data from scattered buffers referenced in sg\_list onto the wire without copying to contiguous memory. If the number of disjoint memory buffers exceeds the limit of NIC's capability to encapsulate all buffers in one RDMA work request, zRPC coalesces the data into a contiguous memory buffer before transmission. This is because sending a single work request (even with a copy) is faster than sending multiple smaller work requests.

**Header-Data Separation.** To implement header-data separation, we specify two scatter-gather elements sge. The first sge points to the packet header's ring buffer, with its size set to the length of the packet header (72 bytes in zRPC). The second sge points to the data ring buffer, with its size set to the MTU minus the packet header's length. Then, submit the WQE using the RDMA API ibv\_post\_recv. By this method, RDMA will store the packet header and data into the designated buffer when receiving data.

#### 5 Evaluation

We run our experiments on a d6515 [3] CloudLab [26] cluster. Each server has two 32-core AMD EPYC ROME 7452 2.35GHz CPUs, with C-States turned off, running Linux 5.04 with Ubuntu 20.04. These servers are connected by dualport Mellanox ConnectX-5 100 Gbps NICs and a 2x100 Gbps Dell Z9264F-ON switch of which MTU = 4200 bytes (RoCE maximum MTU).

#### 5.1 Serialization Performance

**5.1.1** Comparison to Software-Only Serialization. We start by comparing zBuffer to FlatBuffers and Protobuf, the mainstream software serialization libraries. We use a client to send a serialized message to the server, which deserializes, re-serializes the same payload, and returns it to the client. We use a data structure with a single string field, which captures the minimal overhead for serialization. The results are shown in Fig. 9a, zBuffer only needs to generate the corresponding *sg\_array* in the serialization process and restore the position of a pointer in the deserialization process, so the time required is extremely short (25ns total). In addition, because zBuffer does not involve memory copying, it is insensitive to the size of strings, and the time spent under different sizes of string is almost the same.

On the contrary, FlatBuffers and Protobuf need memory copying, encoding and decoding, and it takes  $0.34 \,\mu s$  for Flatbuffers and  $0.57 \,\mu s$  for Protobuf to serialize and deserialize a 1KB string, which is about  $13.6 \times and 22.8 \times longer$  than that

<span id="page-7-0"></span>![](_page_7_Figure_2.jpeg)

![](_page_7_Figure_3.jpeg)

![](_page_7_Figure_4.jpeg)

- (a) Comparison to software-only serialization
- <span id="page-7-1"></span>(b) Metadata size for different object types
- <span id="page-7-2"></span>(c) Serialization overhead of different object types

Figure 9. Serialization overhead of different serialization libraries.

<span id="page-7-3"></span>Table 1. Average cycles taken for zBuffer and Cornflakes.

| Type       | Single     | Tree-1     | Tree-3      | Tree-5      |
|------------|------------|------------|-------------|-------------|
| zBuffer    | 57 cycles  | 104 cycles | 218 cycles  | 747 cycles  |
| Cornflakes | 133 cycles | 312 cycles | 1212 cycles | 5271 cycles |

of zBuffer, respectively. The time spent by Flatbuffers and Protobuf increases significantly with the increase of string length, with a length of 64KB, the cost time of Flatbuffers and Protobuf is an astonishing 168× and 340× longer than that of zBuffer, respectively.

5.1.2 Comparison to Scatter-Gather Coalescing. Cornflakes [\[57\]](#page-12-10) is a serialization library that solely uses NIC scatter-gather to coalesce data. It cannot avoid the overhead of metadata construction. In contrast, zRPC eliminates the overhead of not only data coalescing but also metadata construction. We compare zBuffer with Cornflakes to show the advantage of scatter-gather reflection. We adopt various object types: Single (with only a single bytes filed), and different depths of the tree (e.g., Tree-4 represents a binary tree with nested leaf bytes fields of depth 4).

Fig. [9b](#page-7-1) shows the metadata size under different object types (for Cornflakes, metadata is the object header, for zBuffer it is the object itself). We find that as data structures become more complex, the metadata size of Cornflakes is significantly larger than zBuffer. For Tree-5 type, the metadata size of zBuffer is 512 bytes, while Cornflakes requires 1256 bytes, which is about 2.5× larger than zBuffer. The reason for this is that zBuffer can correctly recognize field information by storing only leaf fields, while Cornflakes requires many sub-headers to help get field information.

Fig. [9c](#page-7-2) shows the serialization and deserialization overhead for different object types. Even with the simplest data types Single, zBuffer is 2.3× faster than Cornflakes. For the more complex object type, Tree-5, zBuffer is 7 × faster than Cornflakes. We also measure the overhead by CPU cycles. The results in Table [1](#page-7-3) demonstrate that zBuffer significantly reduces CPU usage compared to Cornflakes, using only 13.8%

of the CPU cycles required by Cornflakes at Tree-5. The reason is that Cornflakes needs to construct object header as metadata when serializing, and need to read object header to parse field information when deserializing, the more complex the data structure, the higher the overhead. However, zBuffer only needs to send the object itself when serializing, and then it can parse the fields correctly from the object without reading extra object header when deserializing. The overhead of doing this is very small.

### 5.2 End-to-End RPC Performance

5.2.1 zRPC vs. eRPC. We first compare zRPC with the state-of-the-art RPC system, eRPC [\[37\]](#page-11-8), through a set of micro benchmarks. We use two machines, one for the client and the other for the server. We use FlatBuffers and Protobuf as the serialization layer on top of eRPC, represented by eRPC + FB and eRPC + PB, respectively. Unless specified, the RPC request has a byte array, and the response is also a byte array. We adjust the RPC size by changing the array length.

Latency. We evaluate zRPC latency by issuing RPC of different sizes. We test the latency of the small message that can be sent with only one packet and the large message that needs to be sent by multiple packets to demonstrate the performance of zRPC with varying numbers of packets. As shown in Fig. [10a,](#page-8-0) zRPC achieves P99 latency of 5.3 µs for 64B RPC size and 6.5 µs for 2KB RPC size, eRPC + FB adds 0.4 µs and eRPC + PB adds 0.8 µs to the round-trip latency for 1KB RPC size. Fig. [10b](#page-8-1) shows the median latency for large RPC that require multiple packets, as the request size gets larger and more packets are required, non-contiguous data segments become more frequent, resulting in additional copy overhead. zRPC speeds up 3.3× and 4.2× for eRPC + FB and ePRC + PB, respectively, for 1MB RPC size.

Throughput. The client and server in our throughput test use a single application thread and keep 16 concurrent RPCs with different request size. Fig. [10c](#page-8-2) shows the RPC throughput with different request size. zRPC achieves 1.68 Mrps with a request size of 512B, compared to 1.45 Mrps for eRPC + FB and 1.21 Mrps for eRPC + PB, outperforming them by

<span id="page-8-0"></span>![](_page_8_Figure_2.jpeg)

<span id="page-8-1"></span>Figure 10. The latency and throughput vary with the size of RPC requests and responses.

<span id="page-8-3"></span>![](_page_8_Figure_4.jpeg)

<span id="page-8-4"></span>Figure 11. Comparison of RPC scalability.

<span id="page-8-5"></span>![](_page_8_Figure_6.jpeg)

**Figure 12.** Benefits of Header-Data Separation.

15.8% and 38.8%, respectively. As the request size increases, the throughput of all the tested solutions decreases due to being limited to the maximum bandwidth of the RDMA network. The performance of eRPC + FB and eRPC + PB drops drastically to 0.2 Mrps and 0.15 Mrps, respectively. This drop is due to the memory copying required in the serialization process of FlatBuffers and Protobuf, as well as additional memory copies needed for large request sizes that require multiple packets to be sent. In contrast, zRPC maintains a performance of 0.4 Mrps, outperforming eRPC + FB by 2× and eRPC + PB by 2.5×.

Scalability. We evaluate the multicore scalability of zRPC by setting the RPC request size to 1KB and 8KB for testing single packet as well as multi-packet scenarios and increasing the number of client threads. Correspondingly, the server use an equal number of threads, with each client thread connecting to a specific server thread and keeping 16 concurrent RPCs. Fig. 11a shows the RPC throughput when scaling from 1 to 8 user threads with a request size of 1KB. All the tested solutions scale well because at this point the memory copy

<span id="page-8-2"></span>overhead of 1KB size data can be hidden before encountering a network bottleneck. However, as shown in Fig. 11b, when the request size of 8KB is larger than an MTU and requires multiple packet transmission, the throughput of eRPC + FB and eRPC + PB will decrease from 4 to 8 threads, only 1.32Mrps, while zRPC's throughput scales by 1.4×, achieves 1.78Mrps which is 34.8% higher. This is because when the message size becomes larger, eRPC + FB and eRPC + PB consume more CPU cycles during transmission, and the CPU can be easily overwhelmed, which degrades the overall performance. While zRPC implements serialization and deserialization with minimal overhead and further avoids memory copy of eRPC during multiple packet transmission by header-data separation.

**Header-Data Separation.** We demonstrate the performance benefits of Header-Data Separation by setting the RPC client thread (s) from 1 to 8 with 8KB request size. As shown in Fig. 12, zRPC's header-data separation design achieves a 34% throughput improvement with a single thread, entirely eliminating the memory copy overhead associated with multi-packet transmission. However, as the number of threads increases, packets from different clients become interleaved, resulting in additional memory copy overhead. At 8 threads, it achieves a 4.7% throughput improvement. If a dedicated ring buffer is allocated for each connection, memory copies can be completely avoided, but this approach does not scale well. Our design where all client threads share the same ring buffer and traffic control is achieved through the mechanism of credits can work well for both one-to-one and one-to-all communication methods and does not bring performance degradation in other cases.

**5.2.2 zRPC vs. Cornflakes RPC.** Cornflakes ships a codesigned networking stack to realize RPC. To compare zRPC and Cornflakes RPC, we build an echo system in which the client sends a serialized data structure, and the server returns it after deserializing and reserializing, without any additional data handling processes. To generate load, we employ a 16-threaded client that sends concurrent requests. Fig. 13a shows the highest throughput achieved for various

<span id="page-9-0"></span>![](_page_9_Figure_2.jpeg)

<span id="page-9-2"></span>![](_page_9_Figure_3.jpeg)

(a) Achieved throughput for various (b) Throughput-latency tradeoff on a object types. (b) Throughput-latency tradeoff on a custom KV, serving CDN workload.

<span id="page-9-1"></span>**Figure 13.** Comparison between zRPC and Cornflakes.

**Table 2.** Latency comparison for replicated PUTs.

|         | <b>Median Latency</b> | P99 Latency |
|---------|-----------------------|-------------|
| zRPC    | 16.2 μs               | 18.9 µs     |
| eRPC+FB | 18.4 μs               | 21.2 μs     |
| eRPC+PB | 20.5 μs               | 25.1 μs     |

object types with a total size of 1KB. Compared to the single object type, Cornflakes' throughput decreases by 69.5% to 0.46 Mrps in Tree-3. In contrast, zRPC experiences only a 15.6% decline, maintaining a throughput of 1.4 Mrps in Tree-3, which is about 204% higher than Cornflakes. As data structure becomes more complex, more CPU cycles are required to construct the metadata and deserialization also needs to read the metadata for subsequent operations. Furthermore, large metadata will also cause additional network transmission overhead. For example, the metadata size of cornflakes for Tree-3 is 296 bytes, which is ~30% of the actual data size (1024 bytes).

#### 5.3 Real Applications

**5.3.1 Raft.** Raft [51] is a consensus algorithm designed to manage a replicated log in a distributed system, ensuring that multiple servers agree on a shared state even in the presence of failures. A leader is elected among candidates to handle requests from clients. We combine zRPC with LibRaft [70], implement a 3-way replicated in-memory keyvalue store and use one client to issue PUT requests. The client randomly generates a 512-byte key and a 1024-byte value. The client then serializes them and sends them to the leader as a PUT request. The leader receives the request, deserializes it, and gets the contents of the key and value. Note that Cornflakes is implemented in Rust, and its network stack API does not support integration with existing Raft implementations (like raft-rs [10]). We therefore exclude Cornflakes from this experiment. Table 2 shows the PUT latency of the client. zRPC reduces 12% and 21% in median latency and 19% and 24% in P99 latency compared to eRPC + FB and eRPC + PB, respectively. The main takeaway is faster consistent replication is achievable in commodity Ethernet datacenters with zRPC.

<span id="page-9-3"></span>![](_page_9_Figure_11.jpeg)

<span id="page-9-4"></span>Figure 14. P99 latency and throughput for masstree GET.

<span id="page-9-5"></span>![](_page_9_Figure_13.jpeg)

**Figure 15.** Breakdown of CPU cycles in Fig. 13b.

**5.3.2 Key-Value Store.** We use the Masstree KV [46] to evaluate zRPC. Masstree is a high-performance in-memory key-value store with μs-scale service times. We use YCSB-C [22] to generate a trace of 1 million 23-24 byte keys with different size values and use it to populate a Masstree server. We run the Masstree server on one machine and run the client on another machine. Both the server and the client use 8 threads, with each client thread using 16 concurrent requests. The test runs for 60 sec. Fig. 14a shows that zRPC reduces P99 latency by 24.2% and 34.8% compared to eRPC + FB and eRPC + PB, respectively, for 16KB value size. Fig. 14b shows that zRPC achieves 61.9% higher throughput than the eRPC + FB and eRPC + PB. Eliminating copies for larger values can allow the KV to more effectively use the CPU cache for smaller values, thus improving performance.

As Cornflakes cannot support Masstree KV, we use Cornflakes's built-in custom KV to compare zRPC with Cornflakes RPC. Keys are strings and values are lists of DMA-safe buffers. We use the Tragen cache trace generator [59] to create synthetic traces consisting of one million objects using the "image" traffic class, which replicates the traffic patterns observed in Akamai's production CDN. Object sizes range from 1KB to 116MB, with an average size of about 20KB. Since the Cornflakes prototype supports only single-frame messages, we divide each object into MTU-sized (4KB) subobjects. Although each client requests a single sub-object, we report throughput based on the entire object received. Fig. 13b shows the latency-throughput curve on the CDN trace workload. zRPC achieves 33.9% higher than Cornflakes on a p99 latency of 32 µs. Fig. 15 breaks down the average cycles for different parts of request handling within the CDN trace, for the achieved load of about 110K objects per second. Note that "set value" refers to assigning values to each

field and generating an object that the networking stack can directly transmit. zRPC eliminates metadata construction, also enables data processing operations to complete faster. zRPC achieves a 33.6% reduction in CPU cycle consumption compared to Cornflakes, comprising 10.4% fewer metadata construction, 14% fewer networking stacks, and 9.2% fewer deserialization and get/set operations combined.

## 6 Discussion and Related Work

zBuffer with other languages and IDLs. The implementation of zBuffer leverages C++ template metaprogramming and macros, a methodology that is widely used in HPC libraries such as CUTLASS [\[48\]](#page-11-24), OpenBLAS [\[50\]](#page-11-15), and Eigen [\[4\]](#page-10-13). C++ compilers can generate zBuffer object code and expose it through a C ABI, enabling cross-language interoperability. To facilitate the conversion of messages from other Interface Description Languages (IDLs) such as Protobuf and Flat-Buffers into the zBuffer message format, it is necessary to implement a program that parses the corresponding schema files (e.g., .proto and .fbs). This program should establish the mapping relationships required for type conversion and extract the relevant definition blocks and field information for the transformation process. Upon performing these steps, developers can then utilize zBuffer to integrate the converted message structures into their applications.

Zero-copy. zIO [\[65\]](#page-12-16) transparently removes application copies by interposing on memmove and memcpy and handling memory safety via page faults. However, zIO mostly studies packet sizes larger than 8000 bytes and if the buffers are unaligned, a memory copy is still needed. In contrast, zRPC benefits for both small and large packet sizes, and can avoid copying from unaligned buffers. Linux proposes a zero-copy API [\[24\]](#page-11-25), but which is not optimized for µs-scale applications. A wide range of kernel-bypass techniques have been proposed to reduce the overhead imposed by kernel software stacks and context switches caused by system calls, while enabling zero-copy access from the application. For instance, DPDK and RDMA [\[15,](#page-10-7) [35\]](#page-11-12) enable applications to directly access NICs bypassing the kernel.

Optimizing serialization. There have been several attempts to optimize serialization through software improvements. Cap'n proto [\[67\]](#page-12-9) reduce serialization overhead by making the in-memory format match the wire-format exactly. Other approaches include utilizing SIMD parallelism for decoding [\[41\]](#page-11-26) and minimizing the cost of type inference in dynamic serialization [\[47\]](#page-11-27). These approaches do not eliminate the fundamental cost of in-memory copies.

Serialization accelerators. Recent works have proposed specialized hardware for RPC serialization [\[36,](#page-11-28) [40,](#page-11-29) [42,](#page-11-30) [52,](#page-11-31) [53,](#page-11-32) [71\]](#page-12-17). For instance, Zerializer [\[71\]](#page-12-17) offloads serialization logic onto the DMA path, while Cereal [\[36\]](#page-11-28) co-designs serialization formats with hardware architectures.

