# Algorithm 1: Quantize

**Input** :A *D*-dimensional vector **o**′; the number of bits per dimension *B*.

**Output:** The quantization code  $\bar{\mathbf{y}}_u$ .

```
1 \ t \leftarrow 0, v_{max} \leftarrow 0, t_{max} \leftarrow 0
```

<sup>2</sup> Initialize  $\mathbf{y}_{cur}, \langle \mathbf{y}_{cur}, \mathbf{o'} \rangle$  and  $\|\mathbf{y}_{cur}\|$  with t=0

3 while some critical values have not been enumerated do

4 Update *t* with the next smallest critical value

Update  $y_{cur}$ ,  $\langle y_{cur}, o' \rangle$  and  $||y_{cur}||$  with the new t

6 if  $\langle \mathbf{y}_{cur}, \mathbf{o}' \rangle / ||\mathbf{y}_{cur}|| > v_{max}$  then 7  $|v_{max} \leftarrow \langle \mathbf{y}_{cur}, \mathbf{o}' \rangle / ||\mathbf{y}_{cur}||, t_{max} \leftarrow t$ 

8 Compute  $\bar{\mathbf{y}}$  via re-scaling and rounding  $\mathbf{o}'$  with  $t_{max}$ 

<span id="page-4-4"></span>9 **return**  $\bar{\mathbf{y}}_u$  where  $\bar{\mathbf{y}}_u = \bar{\mathbf{y}} + (2^B - 1)/2 \cdot \mathbf{1}$ 

