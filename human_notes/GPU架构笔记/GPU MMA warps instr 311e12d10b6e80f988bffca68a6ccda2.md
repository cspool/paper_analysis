# GPU MMA warps instr

ref：EFFICIENT MATRIX MULTIPLY AND ADD WITH A GROUP OF WARPS

## Fig1A、1B、1C

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image%201.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> | B <sub>0.4</sub> | B <sub>0,5</sub> | B <sub>0,6</sub> | B <sub>0,7</sub> |
> |------------------|------------------|------------------|------------------|
> | B <sub>1,4</sub> | B <sub>1,5</sub> | B <sub>1,6</sub> | B <sub>1,7</sub> |
> | B <sub>2,4</sub> | B <sub>2,5</sub> | B <sub>2,6</sub> | B <sub>2,7</sub> |
> | B <sub>3,4</sub> | B <sub>3,5</sub> | B <sub>3,6</sub> | B <sub>3,7</sub> |
> 
> | A <sub>0,0</sub> | A <sub>0,1</sub> | A <sub>0,2</sub> | A <sub>0,3</sub> |
> |------------------|------------------|------------------|------------------|
> | A <sub>1,0</sub> | A <sub>1,1</sub> | A <sub>1,2</sub> | A <sub>1,3</sub> |
> | A <sub>2,0</sub> | A <sub>2,1</sub> | A <sub>2,2</sub> | A <sub>2,3</sub> |
> | A <sub>3,0</sub> | A <sub>3,1</sub> | A <sub>3,2</sub> | A <sub>3,3</sub> |
> 
> | A <sub>4,0</sub> | A <sub>4,1</sub> | A <sub>4,2</sub> | A <sub>4,3</sub> |
> |------------------|------------------|------------------|------------------|
> | A <sub>5,0</sub> | A <sub>5,1</sub> | A <sub>5,2</sub> | A <sub>5,3</sub> |
> | A <sub>6,0</sub> | A <sub>6,1</sub> | A <sub>6,2</sub> | A <sub>6,3</sub> |
> | A <sub>7,0</sub> | A <sub>7,1</sub> | A <sub>7,2</sub> | A <sub>7,3</sub> |
> 
> | C <sub>0,0</sub> | C <sub>0,1</sub> | C <sub>0,2</sub> | C <sub>0,3</sub> |
> |------------------|------------------|------------------|------------------|
> | C <sub>1,0</sub> | C <sub>1,1</sub> | C <sub>1,2</sub> | C <sub>1,3</sub> |
> | C <sub>2,0</sub> | C <sub>2,1</sub> | C <sub>2,2</sub> | C <sub>2,3</sub> |
> | C <sub>3,0</sub> | C <sub>3,1</sub> | C <sub>3,2</sub> | C <sub>3,3</sub> |
> 
> | C <sub>4,0</sub> | C <sub>4,1</sub> | C <sub>4,2</sub> | C <sub>4,3</sub> |
> |------------------|------------------|------------------|------------------|
> | C <sub>5,0</sub> | C <sub>5,1</sub> | C <sub>5,2</sub> | C <sub>5,3</sub> |
> | C <sub>6,0</sub> | C <sub>6,1</sub> | C <sub>6,2</sub> | C <sub>6,3</sub> |
> | C <sub>7,0</sub> | C <sub>7,1</sub> | C <sub>7,2</sub> | C <sub>7,3</sub> |
> 
> | C <sub>0,4</sub> | C <sub>0,5</sub> | C <sub>0,6</sub> | C <sub>0,7</sub> |
> |------------------|------------------|------------------|------------------|
> | C <sub>1,4</sub> | C <sub>1,5</sub> | C <sub>1,6</sub> | C <sub>1,7</sub> |
> | C <sub>2,4</sub> | C <sub>2,5</sub> | C <sub>2,6</sub> | C <sub>2,7</sub> |
> | C <sub>3,4</sub> | C <sub>3,5</sub> | C <sub>3,6</sub> | C <sub>3,7</sub> |
> 
> | C <sub>4,4</sub> | C <sub>4,5</sub> | C <sub>4,6</sub> | C <sub>4,7</sub> |
> |------------------|------------------|------------------|------------------|
> | C <sub>5,4</sub> | C <sub>5,5</sub> | C <sub>5,6</sub> | C <sub>5,7</sub> |
> | C <sub>6,4</sub> | C <sub>6,5</sub> | C <sub>6,6</sub> | C <sub>6,7</sub> |
> | C <sub>7,4</sub> | C <sub>7,5</sub> | C <sub>7,6</sub> | C <sub>7,7</sub> |
![image.png](GPU%20MMA%20warps%20instr/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 3
![image.png](GPU%20MMA%20warps%20instr/image%201.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_1.jpeg)
> 
> FIG. 4
![image.png](GPU%20MMA%20warps%20instr/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 5C
> 
> FIG. 5A FIG. 5B
![image.png](GPU%20MMA%20warps%20instr/image%203.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 6A
![image.png](GPU%20MMA%20warps%20instr/image%204.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> warp3 (SMEM warp3 (RF) waip3\$ A: 64x16 C/D: 64x64 64xNx16 32 Threads 32 Threads **SMEM** 
> 
> FIG. 6B
![image.png](GPU%20MMA%20warps%20instr/image%205.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_1.jpeg)
> 
> FIG. 7A
![image.png](GPU%20MMA%20warps%20instr/image%206.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 7B
![image.png](GPU%20MMA%20warps%20instr/image%207.png)

layout swizz、

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> R=1, S=0 R=1, S=1 R=1, S=2
> 
> FIG. 8
![image.png](GPU%20MMA%20warps%20instr/image%208.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 9
![image.png](GPU%20MMA%20warps%20instr/image%209.png)