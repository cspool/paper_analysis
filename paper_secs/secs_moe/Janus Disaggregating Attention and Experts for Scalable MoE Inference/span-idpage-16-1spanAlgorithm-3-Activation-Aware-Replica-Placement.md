# <span id="page-16-1"></span>Algorithm 3 Activation-Aware Replica Placement

#### Input:

- $-n_e$ : number of instances, C: capacity per instance
- $-\mathcal{R}$ : set of replicas,  $l_i$ : load of replica  $i, e_i$ : logical expert of replica i **Output:**

```
-P(g): replicas assigned on instance g
  1: Initialize P(g) \leftarrow \emptyset, slots[g] \leftarrow C for all g \in \{1, 2, ..., n_e\}
  2: Initialize x_{e,g} \leftarrow 0 for all experts e and g
  3: Sort replicas \mathcal R in decreasing order of l_i
       for all i \in \mathcal{R} do
             G_i \leftarrow \{g \in G \mid slots[g] > 0 \land x_{e_i,g} = 0\}
  5:
  6:
             if G_i \neq \emptyset then

⊳ Slots feasible

  7:
                  g^* \leftarrow \arg\min_{g \in G_i} \sum_{j \in P(g)} a(i, j)
                  P(g^*) \leftarrow P(g^*) \cup \{e_i\}
  8:
                  slots[g^*] \leftarrow slots[g^*] - 1
10:
                  x_{e_i,g^*} \leftarrow 1
11:
            else
                                         ⊳ No feasible slot; resolve via swapping
                  G_i^{\neg} \leftarrow \{g \in G \mid x_{e_i,g} = 0\} \triangleright \text{Instances without expert } e_i
12:
                  H_i \leftarrow \{h \in G \mid slots[h] > 0\} \triangleright Instances with free slots
13:
                  Find g \in G_i^{\neg}, h \in H_i, and j \in P(g) such that x_{e_i,h} = 0
       # Minimize co-activation load penalty of swapping
                  (g, j, h) \leftarrow \arg\min_{g, j, h} \Delta I(i \rightarrow g, j \rightarrow h)
15:
       # Apply replica swapping
                  P(g) \leftarrow (P(g) \setminus \{j\}) \cup \{i\}
16:
                  P(h) \leftarrow P(h) \cup \{j\}
17:
                  x_{e_i,g} \leftarrow 0, \quad x_{e_i,h} \leftarrow 1, \quad x_{e_i,g} \leftarrow 1
```