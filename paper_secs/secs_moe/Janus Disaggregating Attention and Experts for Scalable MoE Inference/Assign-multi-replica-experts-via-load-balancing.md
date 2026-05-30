# Assign multi-replica experts via load balancing

4: **for all**  $e \in \mathcal{E}$  where R(e) = 1 **do** 

5:  $g \leftarrow$  the unique instance in G(e)

6:  $actRep[e] \leftarrow P(e,g)$ 

7:  $load[g] \leftarrow load[g] + 1$ 

# Assign multi-replica experts via load balancing

8: **for all**  $e \in \mathcal{E}$  where R(e) > 1 **do** 

9:  $g^* \leftarrow \arg\min_{g \in \mathcal{G}(e)} \operatorname{load}[g]$ 

10:  $\operatorname{actRep}[e] \leftarrow P(e, g^*)$ 

11:  $\log[g^*] \leftarrow \log[g^*] + 1$ 

