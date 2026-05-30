# A. Calculating the Number of Passes

We will apply our analysis to attention in Section IV. To illustrate ideas, we first start with a simple pedagogical example, shown in Cascade 1.

$$Y = A_k \times B_k \tag{5}$$

$$Z = Y \times A_k \tag{6}$$

Cascade 1: An example 2-pass cascade.

Einsum 5 performs a dot product between  $A_k$  and  $B_k$ , and Einsum 6 multiplies the first Einsum's result Y by  $A_k$  again to produce Z. If we want to minimize data traffic of  $A_k$ , we need to choose a dataflow for each Einsum that keeps  $A_k$  stationary and fuses the two Einsums together. In other words, the dataflow must finish using the first element of  $A_k$  before moving onto the next. However, such a dataflow does not exist for this cascade. Any implementation must visit *every* element of  $A_k$  to compute Y before it can revisit *any* element of  $A_k$  to compute Z.

We define a pass that a cascade performs over a particular fiber of a particular rank and tensor to be a traversal of every element of that fiber. Each time an element must be revisited after visiting every other element of that fiber, there is an additional pass. For example, Cascade 1 performs two passes over the K rank of  $A_k$ .

Since an Einsum's iteration space can also be represented as a fibertree (i.e., an *is-fibertree* – see Section II-B), we extend our definition of an iteration space for a cascade of Einsums by considering its iteration space to be the sequence of the is-fibertrees for each Einsum. Now, in a scenario where fibers for a particular rank exist in multiple is-fibertrees; in each, they project to the same tensor; and there is a dependency such that all of the elements of the earlier is-fibertree's fiber must be read before any element can be read again by the later is-fibertree (for all mappings of the cascade), we refer to that read-read sequence as creating an additional *pass*. When there is a sequence of N such read-read dependencies, we say the cascade is an (N+1)-pass cascade. For our example, Cascade 1 requires two passes of the K rank.

