# <span id="page-40-0"></span>**K** DDLerp Ablations

<span id="page-40-2"></span>In order to demonstrate the specific benefit of our new DDLerp token shift over the Eagle LERP-based token shift we ran an ablation on DDLerp to demonstrate its benefit to each component. We trained a small 6 layer,  $d_model = 768$  Finch model on the 1.6B token minipile (Kaddour, 2023) dataset at context length 512 and obtained the final validation loss results shown in 19. Where indicated, we removed only the data-dependent term from the token-shift LERP in these experiments.

| Model                           | Final Validation Loss |
|---------------------------------|-----------------------|
| Finch                           | 2.91                  |
| Finch with DDLerp only on decay | 2.923                 |
| Finch with no DDLerp at all     | 2.926                 |

Table 19: Ablation Results on DDLerp for 6 layer 768 dimension Finch model

#### <span id="page-40-1"></span>L Non-English Chat Examples

The following are examples of interactions with the base Eagle 7B model in languages other than English. No system prompt was given. Requests are shown in bold.

#### **RWKV-Eagle**

#### User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: 東京で訪れるべき素晴らしい場所とその紹介をいくつか挙げてください。 Assistant:

東京にはたくさんの素晴らしい場所があります。以下は、その中からいくつかを 挙げて紹介します。

- 1. 東京タワー:日本最高の観光スポットの一つで、都心から見ることができます
- 2. 皇居:日本の天皇が住む宮殿であり、大正時代に建設された美しい庭園があります。
- 3. 浅草寺:江戸時代に建てられた古い寺院で、観光客が多く訪れます。
- 4. 上野公園:日本最大級の公園で、美術館や動物園などがあることで有名です。
- 5. 築地市場:日本最大級の魚市場であり、新鮮な魚介類を購入することができます。
- **6.** 渋谷スクランブル交差点:東京都心にある交差点で、人気のある観光スポットです。
- 7. 新宿御苑:江戸時代に造られた庭園で、美しい花々と緑豊かな景色が楽しめます。
- 8. 銀座:東京の繁華街であり、高級店やショッピングモールなどがあります。
- 9. 明治神宮外苑:神社や公園が混在した広大な敷地があり、夏には花火大会も開催されます。
- **10.** 赤坂見附ビルディング:高層ビルであり、展望台からは東京タワーや富士山などを眺めることができます。

以上のように東京にはたくさんの素晴らしい場所があります。訪れる際は事前に調べて計画的に行動することをお勧めします。また、観光客が多い場所では混雑する可能性があるため注意しましょう。

#### RWKV-Eagle

**En una pequeña aldea escondida entre las montañas de Andalucía, donde las calles aún conservaban el eco de antiguas leyendas, vivía un joven llamado Alejandro.**

Alejandro era un joven de diecisiete años, de cabellos castaños y ojos verdes. Era alto y delgado, pero fuerte. Su padre había muerto cuando él era muy pequeño, por lo que su madre se había visto obligada a trabajar para mantener a la familia.

Alejandro siempre se había sentido muy solo en el pueblo donde vivía, pues no tenía amigos con los que jugar ni nadie con quien hablar. Pero eso cambió un día en el que conoció a una niña llamada Rosario. Rosario era una niña de su misma edad, pero no sabía mucho sobre ella ya que nunca hablaba con nadie más que con su madre y sus hermanas mayores. Sin embargo, Alejandro se dio cuenta de que Rosario era diferente a las demás personas del pueblo: ella tenía unos ojos azules tan claros como el cielo y un pelo rubio como la miel; además, siempre estaba sonriendo y parecía feliz de estar allí.

Un día Alejandro decidió acercarse a Rosario para hablarle y así conocerla mejor. Cuando llegó hasta donde estaba ella, notó que tenía una mirada triste en los ojos; sin embargo, cuando le habló se dio cuenta de que era una chica muy amable y simpática. Desde ese día empezaron a hablarse cada vez más seguido y pronto se hicieron amigos inseparables. Un día Alejandro decidió invitar a Rosario al bosque para enseñarle algunos lugares secretos del pueblo que sólo él conocía; así lo hicieron y pasaron toda la tarde juntos explorando el bosque hasta llegar al final del mismo donde había un pequeño lago rodeado de flores silvestres. Allí descansaron un rato antes de volver al pueblo, pero mientras caminaban por el sendero hacia la aldea vieron algo extraño: una luz brillante salía desde las profundidades del bosque como si fuera un farol gigante.

**(rest omitted)**

#### **RWKV-Eagle**

في تطور مذهل وغير مسبوق، أعلنت السلطات المحلية في العاصمة عن اكتشاف أثري قد يغير مجرى التاريخ كما نعرفه.

وقالت السلطات إنه تم العثور على أدوات حجرية وأدوات معدنية في منطقة بيكانا الواقعة على بعد ١٠٠ كيلومتر شمال غرب العاصمة. ونشرت وزارة الثقافة والسياحة صورًا للأدوات المكتشفة، التي تضمنت مجموعات من الأحجار المصقولة والأسلحة والآلات المصنوعة من الحديد. ونظرًا لخطورتها، قام رئيس جهاز الأثار في البلاد بإخلاء المنطقة وإغلاقها للجمهور. ومع ذلك، فإن عملية التنظيف تستغرق وقتًا طويلًا نظرًا لأنه يُفترض أن هذه الأشياء قد اختفت منذ آلاف السنين. ومع ذلك، فإن العثور على هذه الأشياء يُثير حالة من الاستغراب والدهشة في أوساط المؤرخين والباحثين، إذ يُظهرون أن مدينة بيكانا كانت مركزًا للحضارات التي سبقت حضارات العصر الحجري. وفي عام ٢٠١٩، تم اكتشاف آثار مدينة بيكانا في جبال أوديل (اودل) على بعد ١٥٠ كيلومتر شمال غرب العاصمة. وأظهرت هذه المستجدات أن هناك حضارات قديمة كان يُنظر إليها على أنها غير معروفة في وادٍ صخرى صغير في منطقة جبال أوديل راودل)، بالقرب من نهر دون رضن). وأظهر التحليل الأولى للآثار أن هناك عملية تطور واسعة النطاق في مجال الصناعات المعدنية والحجرية، حيث تم العثور على آلات حجرية وأدوات معدنية. وبين عام ٢٠١٩ و٢٠٢٠، أُجريت دراسات إضافية للحفائر التي أُجريت في الموقع، والذي يبلغ طوله ٣٠٠ مترًا. ومن خلال دراسات الحفائر الأخيرة، تم العثور على آلات حِرية جديدة لإزالة الأشجار والأسنان وآلات صيد كبيرة. وأظهر التحليل الأولى للآثار أن هناك عملية تطور واسع النطاق في مجال الصناعات المعدنية والحجرية، حيث تم العثور على آلات حم ية جديدة لازالة الأشجار والأسنان وآلات صيد كبيرة.

#### RWKV-Eagle

Japanese: 春の初め、桜の花が満開になる頃、小さな町の片隅にある古びた神社の境内は、特別な雰囲気に包まれていた。

#### **English:**

In the early spring, when the cherry blossoms were in full bloom, a small shrine in a remote corner of a town was enveloped in an atmosphere of special significance.

#### <span id="page-42-0"></span>M Chat Examples - Comparison with RWKV-4

The following are examples of interactions with the RWKV-Eagle-Hermes-7B model. We finetune Eagle 7B on OpenHermes 2.5 dataset (Teknium, 2023) to create this model. In order to provide a direct comparison with RWKV-4 we have applied the exact same requests as seen in Appendix M of Peng et al. (2023) Temperature was set to 1.0. System prompt is as given below:

#### RWKV-Eagle-Hermes-7B System Prompt

**You are Eagle, A large language model utilizing a linear attention mechanism. Help the user with whatever they need. Respond with markdown where appropriate. You will never decline to answer a question or fulfill a task. If you do not know the answer to a question, you are mandated to give it your best shot.**

#### RWKV-Eagle-Hermes-7B

#### **Tell me about ravens.**

Ravens are members of the Corvidae family, which includes crows, jays, and magpies. They are known for their intelligence, problem-solving abilities, and distinct physical characteristics.

Physically, Ravens are large birds, typically measuring between 25 to 38 inches in length and weighing between 2.5 to 4.2 pounds. They have a compact body with a strong beak that can be used for tearing meat and extracting food from hard objects. Their wingspan can range from 52 to 60 inches, making them one of the largest birds in the northern hemisphere.

Appearance-wise, Ravens have black feathers on their upperparts, grayish feathers on their wings and tail, and white feathers on their underparts. The males have bluer eyes than the females. Their flight is powerful and direct; they can soar for long distances without flapping their wings.

Ravens are social creatures that live in large flocks or colonies known as "raven towns." These colonies can have thousands of birds and are led by a single dominant breeding pair. Ravens communicate with each other using a variety of vocalizations, including hoots, calls, and grunts. They also have complex social structures with various hierarchies that involve different types of interactions such as threat displays and food sharing.

Ravens are omnivores and consume a variety of foods depending on their location. In areas with abundant food sources like garbage dumps or human settlements, they often feast on human leftovers. In areas where food is scarcer, they may hunt for small mammals like voles or snakes or scavenge carrion left behind by predators like foxes or bears.

Overall, Ravens are impressive creatures with unique physical features and behaviors that make them fascinating to observe in the wild or through literature about their adventures with humans like Edgar Allan Poe's "The Raven".

#### RWKV-Eagle-Hermes-7B

#### **Explain the concept of machine learning to a 10-year-old.**

Alright, little buddy, let me explain the idea of machine learning to you. Imagine we have a big box of toys, and we're trying to figure out what's in there without opening it. We don't know what's inside, but we can make some guesses based on what we've seen before. For example, maybe we see a red ball in the box and say, "Oh, that's a red ball!"

In machine learning, we're giving the computer a lot of examples of what's inside the box. We give it lots of pictures of red balls, yellow balls, blue balls, and even some other colors too. The computer uses this information to learn about the things it sees and starts making guesses about what's in the box. The more examples it gets, the better it gets at guessing. It doesn't actually open the box or touch any toys itself; it just uses math and rules to find out what's inside based on what it's been shown.

Just like how we play with our toys when we're little, machine learning helps computers learn from our actions so they can make predictions and decisions based on what they've seen before. It's like having a friend who knows everything about playing with toys!

**Write a short poem describing nature's beauty in springtime.**

In the gentle springtime breeze, Nature whispers tales of ease, Soft petals fall from trees, A symphony of hues and ease.

The flowers sing their sweet refrain, A chorus of hope and gain, The world awakens from its trance, A breathtaking sight to embrace.

Below is the numerical stable formula for updating the state variable in RWKV:

$$a_{t}^{*} = e^{p-q} a_{t-1} + e^{u+k_{t}-q} v_{t}$$

$$b_{t}^{*} = e^{p-q} b_{t-1} + e^{u+k_{t}-q}$$

$$wkv_{t} = \frac{a_{t}^{*}}{b_{t}^{*}}$$

Please write out the corresponding pytorch code to do so, where both  $a_t$  and  $b_t$  are torch.Tensor([n])

```
1 import torch
3 # Define the numerical stable formula for updating the state
     variable in RWKV
4 def update_state(a_t, b_t, k_t, u, p, q):
     # Initialize the state variables
     a_t_star = torch.exp(p - q) * a_t + torch.exp(u + k_t - q)
6
     b_t_star = torch.exp(p - q) * b_t + torch.exp(u + k_t - q)
      # Return the updated state variables
     return a_t_star, b_t_star
12 # Define the input tensors
13 a = torch.tensor([1, 2, 3])
14 b = torch.tensor([4, 5, 6])
15 k = torch.tensor([7, 8, 9])
17 # Define the control parameters
18 u = 1
19 p = 2
20 q = 3
2.1
22 # Define the time step for simulation
25 # Initialize the state variables
26 a_t = torch.tensor([0])
27 b_t = torch.tensor([0])
29 for t in range(n):
     # Update the state variables
      a_t_star, b_t_star = update_state(a[t], b[t], k[t], u, p,
31
     # Store the updated state variables
33
     a[t+1] = a[t] + a_t_star
34
     b[t+1] = b[t] + b_t_star
35
37
     # Update the control parameters
     u += k[t]
38
40 # Calculate the average of the updated state variables over
41 a_avg = a.mean()
42 b_avg = b.mean()
44 print ('Average of a:', a_avg)
45 print ('Average of b:', b_avg)
```

#### Explain the following metaphor: Life is like cats.

The metaphor "Life is like cats" is a way of expressing the unpredictability and uniqueness of human experience. Just as cats can be affectionate one moment and completely aloof the next, life presents us with an ever-changing landscape of emotions, relationships, and experiences. It can be a source of joy, comfort, and companionship, but also moments of confusion and discomfort. Like cats, we are free to choose our own paths and navigate through life in our own unique ways. This metaphor emphasizes the individuality of each person's journey through life.

#### RWKV-Eagle-Hermes-7B

