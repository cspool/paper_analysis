# <span id="page-14-0"></span>C Examples in Training

The question-answer format and size of the datasets in each training stage can be found in Table [3.](#page-14-1)

### C.1 Examples in SFT Stage

We provide four reasoning formats for the SFT dataset AQuA-Rat [\[26\]](#page-10-8), including *Direct Answer*, *Short CoT*, *Code*, *Long CoT*. Here is an example.

```
✞ ☎
Question :
You collect baseball cards . Suppose you start out with 15. Maria takes
 half of one more than the number of baseball cards you have . Since
you ' re nice , you give Peter 1 baseball card . Since his father makes
baseball cards , Paul decides to triple your baseball cards . How many
baseball cards do you have at the end ?
Direct Answer :
18
Code :
def calculate_baseball_cards () :
    # Initial number of baseball cards
    initial_cards = 15
    # Maria takes half of one more than the number of baseball cards
    maria_takes = ( initial_cards + 1) / 2
    cards_after_maria = initial_cards - maria_takes
    # Peter takes 1 baseball card
```

```
cards_after_peter = cards_after_maria - 1
    # Paul triples the number of baseball cards
    final_cards = cards_after_peter * 3
    return {
        ' maria_takes ': maria_takes ,
        ' cards_after_maria ': cards_after_maria ,
        ' cards_after_peter ': cards_after_peter ,
        ' answer ': final_cards
    }
>>> calculate_baseball_cards ()
output = {
    ' maria_takes ': 8.0 ,
    ' cards_after_maria ': 7.0 ,
    ' cards_after_peter ': 6.0 ,
    ' answer ': 18.0
}
Short CoT :
Start with 15 baseball cards . Maria takes half of one more than the
number of baseball cards you have . So Maria takes half of 15 + 1 which
 is 8 , so you ' re left with 15 - 8 = 7. Peter takes 1 baseball card
from you : 7 - 1 = 6 baseball cards . Paul triples the number of
baseball cards you have : 6 * 3 = 18 baseball cards . So you have 18 at
the end . The answer is 18.
Long CoT :
Okay , let me try to work through this problem step by step . So , the
question is about collecting baseball cards , and there are a series of
 steps that happen . Let me read it again to make sure I understand
each part .
Alright , starting with 15 cards . Then Maria takes half of one more
than the number I have . Wait , that phrasing is a bit tricky . So Maria
takes half of ( one more than my current number of cards ) . Let me parse
 that . If I have N cards , Maria takes ( N +1) /2? Hmm . Let me double -
check . The problem says : " Maria takes half of one more than the number
 of baseball cards you have ." So , the expression would be ( ( number
you have + 1 ) ) / 2. Right . So that ' s ( current cards +1) divided by
two . Then after that , I give Peter 1 card . Then Paul triples my cards .
 So , let ' s break it down step by step .
Starting with 15 cards . Let ' s note each step :
1. Initial cards : 15.
2. Maria takes half of one more than what I have . So , first , calculate
 one more than my current count : 15 +1 =16. Then she takes half of
that . So 16 /2 =8. So Maria takes 8 cards . Wait , but when she takes
them , does that mean my total decreases by 8? Yes , I think so . Because
 she ' s taking them from me . So subtract 8 from my current count .
So after Maria takes them : 15 -8 =7. Now I have 7 cards left .
Wait a minute , wait . Wait , when Maria takes half of one more than the
number I have , which is (15+1) /2=8. So she ' s taking 8 cards . So yes ,
subtract 8 from my current 15. That leaves me with 7.
Then , the next step : since I ' m nice , I give Peter 1 baseball card . So
subtract 1 more . 7 -1 =6.
```

```
Then Paul decides to triple my baseball cards . So multiply my current
count by 3. 6 *3 =18.
So the answer should be 18. Let me go through the steps again to check
.
Original : 15.
After Maria : (15 +1) /2 =8. So 15 -8=7.
After giving Peter 1: 7 -1=6.
After Paul triples : 6*3=18.
Yes , that seems right . Did I misinterpret any step ? Let ' s double - check
 each part .
" Maria takes half of one more than the number of baseball cards you
have ." So when Maria takes , it ' s ( my current cards +1) /2. So when I
have 15 , that ' s (15+1) /2=8. She takes 8 , so I have 7 left . Correct .
Then giving Peter 1: 7 -1=6. Then Paul triples to 18. Yep . That seems
correct . So the final answer is 18.
✝ ✆
```

#### C.2 Examples in RL Stage

The RL dataset includes CSQA [\[44\]](#page-11-8), GSM8K [\[6\]](#page-9-10), and MATH [\[15\]](#page-10-10), with examples provided below.

```
✞ ☎
< CommonsenseQA >
What is the base of a vase called ?
( A ) transistor ( B ) vessel ( C ) box ( D ) pie ( E ) seabase
ground truth : B
< GSM8K >
Natalia sold clips to 48 of her friends in April , and then she sold
half as many clips in May . How many clips did Natalia sell altogether
in April and May ?
ground truth : 72
< MATH >
Rationalize the denominator : $\ frac {1}{\ sqrt {2} -1} $ . Express your
answer in simplest form .
ground truth : $ \ boxed {\ sqrt {2}+1} $
✝ ✆
```

