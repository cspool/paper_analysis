# **G.1 HTML to TSV**

Example [G.1](#page-34-0) shows an example of GPT-4o failing to copy all rows from the HTML input of an 8K test set without filters. Note that the model is only able to extract the first 5 rows and then skip the remaining rows.

### <span id="page-34-0"></span>Example G.1: An example of copying error without filters for HTML to TSV **HTML Page** <html><head> <title>January 2021 - Page 2 of 2 - Juicers Zones</title> </head> <body> <div> <a>Skip to content</a> <header> <div> <div> <div> <div> <div> <h3> <a title="Juicers Zones">Juicers Zones</a> </h3> </div></div><div> <nav role="navigation"> <p> <span>Menu</span> </p> <div><ul><li><a>Juice Zone</a></li> <li><a>Privacy Policy</a></li> <li><a>Contact</a></li> </ul></div></nav> </div></div></div></div><div><img alt="Juicers Zones"/></div> <div> <div> <div> <h1>Month: <span>January 2021</span></h1> </div> </div> </div> </header> <div> <div> <div> <div> <article> <header> <h2> <a title="Lidar Pasadena..">Lidar Pasadena..</a> </h2></header> <div> <p>Local Surveyor Pasadena Hiring a land surveyor is something many people do only once or twice in their lives, so they don't have plenty of experience when determining who to hire. Hiring a surveyor is, in many ways, like hiring</p> </div> <footer><div> <span><a>Sheree</a></span> <span><a title="5:01 pm"><time>January 14, 2021</time></a></span><span><a>Juice Zone</a></span> <span> <a>Read more</a>... ... </body> ... ... </html> **Target Information** Based on the HTML webpage above about Articles, extract the following properties from the items

listed on the webpage: (1) Title; (2) Author; (3) Date; (4) Description;

### **Ground Truth TSV**

Lidar Pasadena.. Sheree January 14, 2021 Local Surveyor Pasadena Hiring a land surveyor is something many people do only once or twice in their lives, so they don't have plenty of experience when determining who to hire. Hiring a surveyor is, in many ways, like hiring

Happy Valentines Day Wishes – Remarkable Benefits.. Sheree January 14, 2021 Valentine's Day is recognized every year by people and allows men and women to show the love they have for their beloved. Even though all have special days, this particular day is accepted as being a special day to espouse

WWE Corporate Office Headquarters – Why Is This Significant.. Sheree January 12, 2021 The WWE Divas are some of the hottest ladies on the planet today, as well as many of the most athletic entertainers. The latest divas include the likes of Candice Michelle, Torrie Wilson, Melina Perez, Maria Kanellis, Victoria, Jillian Hall

9 Skincare Tips – Check This Out.. Sheree January 11, 2021 Beauty Tips Blog Taking care of your body means getting regular exercise, and good nourishment nevertheless in the rushed fast paced lifestyles in our 21st century, and many holding down demanding careers, this may not be quite simple to do.

Paraiba World Ltd – Just Published Sheree January 9, 2021 Automated buying and selling platforms driven by configurable crypto buying and selling crawlers are increasing. Exactly what are the pros and cons of those platforms, that are the marketplace leaders – and must you make use of one? The maturing

Kroger Headquarters Contacts – Useful Information.. Sheree January 8, 2021 You can either get your Kroger weekly advertisements every weekend break and let it sit there unused or you can put it to great use and really start saving lots of money. Me, I prefer to make the most of

... ...

[43 more rows]

... ...

### **Model Output TSV**

Lidar Pasadena.. Sheree January 14, 2021 Local Surveyor Pasadena Hiring a land surveyor is something many people do only once or twice in their lives, so they don't have plenty of experience when determining who to hire. Hiring a surveyor is, in many ways, like hiring

Happy Valentines Day Wishes – Remarkable Benefits.. Sheree January 14, 2021 Valentine's Day is recognized every year by people and allows men and women to show the love they have for their beloved. Even though all have special days, this particular day is accepted as being a special day to espouse

WWE Corporate Office Headquarters – Why Is This Significant.. Sheree January 12, 2021 The WWE Divas are some of the hottest ladies on the planet today, as well as many of the most athletic entertainers. The latest divas include the likes of Candice Michelle, Torrie Wilson, Melina Perez, Maria Kanellis, Victoria, Jillian Hall

9 Sk

Example [G.2](#page-35-0) shows an example of GPT-4o extracting wrong rows that do not satisfy the given filtering instruction from a 2K test example with filters. While the model is prompted to extract rows where the type of the household is strictly equal to "2 Beds" (partial match such as "1-2 Beds" is not valid), the model output still includes rows where the type of the household is "Studio" or "1-3 Beds".

