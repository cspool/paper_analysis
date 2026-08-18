# B. Step-2: Balancing Clusters

Naive clustering by KMeans can still leave imbalanced and oversize clusters, whose member sets are larger than the list size r we set. Here, we split these overly large clusters until no cluster has more than r members. Now, we describe the procedures with the help of Algorithm 2. For a cluster with more than r members, we split it into smaller clusters using KMeans (line 5). We repeat this procedure until all clusters have fewer than r members (lines 2-14). One tricky case here is handling the candidates. When splitting, we choose to let child clusters directly inherit the candidates from the parent cluster for the benefit of overall search efficiency (lines 6-9). This is because sibling clusters are close to each other and thus likely to be queried and loaded to memory together. If we stick to the candidate selection in Step-1, sibling clusters would add each other's members to form a majority of the candidates, leading to redundant padding. Consequently, due to the close distance, sibling clusters are likely to be loaded to memory together during search. This means the same vectors (i.e.,

## Algorithm 2 Balancing clusters by splitting oversize ones

```
Input: initial centroids C_{init}, member sets I, candidate sets E, list size r
 Output: new centroids C_{bala}, new member sets I', new candidate sets E
 1: C_{bala}, I', E' \leftarrow \emptyset
     while C_{init} \neq \emptyset do
 2:
 3:
          Select c \in C_{init}
          if sizeof(I[c]) > r then
              C_{sub}, M_{sub} \leftarrow KMeans(I[c], \lceil \frac{sizeof(I[c])}{r} \rceil)
 5:
 6:
              for c_{sub} \in C_{sub} do
 7:
                   C_{init} \leftarrow C_{init} \cup \{c_{sub}\}, I[c_{sub}] \leftarrow M[c_{sub}]
 8.
                   E[c_{sub}] \leftarrow E[c]
 9.
              end for
10:
              \begin{array}{l} C_{bala} \leftarrow C_{bala} \cup \{c\}, \ I'[c] \leftarrow I[c], \ E'[c] \leftarrow E[c] \\ \text{Remove } c \ \text{from } C_{init}, \ I, \ E \end{array}
11:
12:
13:
           end if
14: end while
15: return C_{bala}, I', E'
```

redundant candidates) will be repeatedly loaded, impacting overall search efficiency.

# B. Step-2: Balancing Clusters

Naive clustering by KMeans can still leave imbalanced and oversize clusters, whose member sets are larger than the list size r we set. Here, we split these overly large clusters until no cluster has more than r members. Now, we describe the procedures with the help of Algorithm 2. For a cluster with more than r members, we split it into smaller clusters using KMeans (line 5). We repeat this procedure until all clusters have fewer than r members (lines 2-14). One tricky case here is handling the candidates. When splitting, we choose to let child clusters directly inherit the candidates from the parent cluster for the benefit of overall search efficiency (lines 6-9). This is because sibling clusters are close to each other and thus likely to be queried and loaded to memory together. If we stick to the candidate selection in Step-1, sibling clusters would add each other's members to form a majority of the candidates, leading to redundant padding. Consequently, due to the close distance, sibling clusters are likely to be loaded to memory together during search. This means the same vectors (i.e.,

## Algorithm 2 Balancing clusters by splitting oversize ones

```
Input: initial centroids C_{init}, member sets I, candidate sets E, list size r
 Output: new centroids C_{bala}, new member sets I', new candidate sets E
 1: C_{bala}, I', E' \leftarrow \emptyset
     while C_{init} \neq \emptyset do
 2:
 3:
          Select c \in C_{init}
          if sizeof(I[c]) > r then
              C_{sub}, M_{sub} \leftarrow KMeans(I[c], \lceil \frac{sizeof(I[c])}{r} \rceil)
 5:
 6:
              for c_{sub} \in C_{sub} do
 7:
                   C_{init} \leftarrow C_{init} \cup \{c_{sub}\}, I[c_{sub}] \leftarrow M[c_{sub}]
 8.
                   E[c_{sub}] \leftarrow E[c]
 9.
              end for
10:
              \begin{array}{l} C_{bala} \leftarrow C_{bala} \cup \{c\}, \ I'[c] \leftarrow I[c], \ E'[c] \leftarrow E[c] \\ \text{Remove } c \ \text{from } C_{init}, \ I, \ E \end{array}
11:
12:
13:
           end if
14: end while
15: return C_{bala}, I', E'
```

redundant candidates) will be repeatedly loaded, impacting overall search efficiency.

