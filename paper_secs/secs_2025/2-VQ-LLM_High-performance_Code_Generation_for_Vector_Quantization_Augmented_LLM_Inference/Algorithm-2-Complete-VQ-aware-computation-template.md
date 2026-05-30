# **Algorithm 2** Complete VQ-aware computation template

**Input:** quantized, codebook, compute\_op

```
Output: output
 1: function Kernel_Template
         All, Reduce \leftarrow compute\_op.all\_axes, reduce\_axes
3:
         layout_{src.dst} \leftarrow codebook.vector\_size, compute\_op.required\_size
         Budget \leftarrow Free shared and reg to preserve occupancy
 4:
         factor \leftarrow Value \ to \ make \ Traffic_{Reduce} = Traffic_{Codebook}
5:
         n_{\textit{shuffle}} \leftarrow layout_{\textit{src}}/layout_{\textit{dst}}
 7:
         if n_{shuffle} \leq thres_{shuffle} = 5 then
8:
           Thread_Mapping(compute_op.warp_tile, layout_src.dst)
         {\tt Parallel\_For}(codebook.switch\_axes, factor)
10:
            if required by algorithm then
               CB \leftarrow \textbf{Switch}(New\ codebook\ ptr)
11.
12.
            CB_{cached}, boundry \leftarrow \texttt{Load}(CB, Budget)
13:
            for id in quantized_data do
               data \leftarrow \texttt{Access}(CB_{cached}, boundry, CB, id)
14:
15:
            if n_{\textit{shuffle}} \leq thres_{\textit{shuffle}} then
16.
               data \leftarrow \texttt{Reg\_Fusion}(data, n_{\textit{shuffle}})
17:
18:
               data \leftarrow \texttt{Shared\_Fusion}(data, layout_{src, dst})
            for temporal\_iteration on All-codebook.switch\_axes do
20:
               partial \leftarrow compute \ op(data, temporal \ iteration)
21:
            output \leftarrow \textbf{Reduce}(partial, Reduce \cap codebook.switch\_axes)
         Return output
```

