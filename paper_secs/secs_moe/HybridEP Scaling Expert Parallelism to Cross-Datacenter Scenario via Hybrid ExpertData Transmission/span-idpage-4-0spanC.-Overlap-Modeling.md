# <span id="page-4-0"></span>C. Overlap Modeling

Overlap Modeling of Two Streams. The overlap time between computation and communication comes from three parts, as shown in Figure 5(a) split by the red dotted line. Specifically, ① pre-expert computation (i.e.,  $Lat_{comp}^{PE}$ ) and AG; ② expert computation ( $Lat_{comp}^{Ep}$ ) and AG; ③ expert computation ( $Lat_{comp}^{Ep}$ ) and A2A; Note that the pre-expert computation cannot overlap with A2A because the data depend on the pre-expert results. Therefore, overlap can be written as:

$$Lat_{ovlp} = Lat_{ovlp}^{PE,AG} + Lat_{ovlp}^{Ep,AG} + Lat_{ovlp}^{Ep,A2A}, \quad (6)$$

Note that case ② and ③ have been optimized by previous works [35], [46], therefore expert computation is fully overlap with AG and A2A ( $Lat_{ovlp}^{Ep,AG} + Lat_{ovlp}^{Ep,A2A} = nLat_{comp}^{Ep}$ ). For case ①, the overlap time is  $min(Lat_{comp}^{PE}, Lat_{comm}^{AG})$ . Therefore, the final overlap time can be expressed as:

<span id="page-4-4"></span>
$$Lat_{outp} = min(Lat_{comn}^{PE}, Lat_{comm}^{AG}) + nLat_{comn}^{EP}, \quad (7)$$

