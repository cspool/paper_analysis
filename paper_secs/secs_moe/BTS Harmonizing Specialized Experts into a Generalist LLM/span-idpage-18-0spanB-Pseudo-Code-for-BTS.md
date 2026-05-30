# <span id="page-18-0"></span>B Pseudo Code for BTS

```
d e f S ti t c h L a y e r ( xs , merge_into_hub=True ) :
    """
     xs : dense models ' outputs
    """
     x_hub = x [ 0 ]
     x_experts = x [ 1 : ]
     g = w_gate ( x_hub ) # [ bs , seq_len , dim, 1+n_experts ]
    # Experts - into -Hub Layer
     i f merge_into_hub :
          g = dropout ( g ) . so ftmax ( dim=- 1 )
          h_experts = [
               w_proj [ i ] ( x_experts [ i ] ) f o r i i n r an ge ( n_experts )
          h_hub = ( g * s t a c k ( [ h ] + h_experts , dim=- 1 ) ) . sum( - 1 )
    # Merge- into -Expert Layer
     e l s e :
          g = dropout ( g ) . si gm oid ( )
          h_experts = = [
                     ( 1 - g [ . . . , i + 1 ] ) * x_experts [ i ]
                    + ( g [ . . . , i + 1 ] * w_proj [ i ] ( x_hub ) )
                     f o r i i n r an ge ( n_experts )
          h_hub = x_hub
     r e t u r n s t a c k ( [ h_hub ] + h_experts , dim=- 1 )
d e f BTSBlock ( xs , i t h_l a y e r , BTS_freq ) :
     x_hub = hub_model_layer ( xs [ 0 ] )
     x_experts = [ expert_model_layer [ i ] ( xs [ i+1 ] ) f o r i i n n_experts ]
     xs = s t a c k ( [ x_hub ] + x_experts , dim=- 1 )
     i f i t h_l a y e r % BTS_freq = = 0 :
         # Alternate between two types of s t i t ch layers
          hs = S ti t c h L a y e r ( xs , merge_into_hub=( i t h_l a y e r // BTS_freq )%2 )
          r e t u r n hs
     r e t u r n xs
```