# <span id="page-11-0"></span>F Details of MeetingBank QA and MeetingBank Summary

The MeetingBank QA dataset consists of 862 meeting transcripts from the MeetingBank test set. Initially, we generate 10 question-answer pairs for each meeting transcript using GPT-4-32K. The instruction used in generating QA pairs is: "*Create 10 questions/answer pairs from the given meeting transcript. The answer should be short and concise. The question should start with Q: and answser should start with A: . The meeting transcript is as follows.*". To ensure the quality of the generated QA pairs, we discard the question-answer pairs with answer lengths exceeding 50 tokens. Subsequently, we carefully examine the remaining QA pairs to ensure that the answers actually appear in the original transcripts, instead of being products of GPT-4's hallucinations. After the aforemen-

<span id="page-11-3"></span><sup>5</sup> https://spacy.io/api/lemmatizer

## <span id="page-12-1"></span>Original Prompt (200 Tokens):

Item 15, report from City Manager Recommendation to adopt three resolutions. First, to join the Victory Pace program. Second, to join the California first program. And number three, consenting to to inclusion of certain properties within the jurisdiction in the California Hero program. It was emotion, motion, a second and public comment. CNN. Please cast your vote. Oh. Was your public comment? Yeah. Please come forward. I thank you, Mr. Mayor. Thank you. Members of the council. My name is Alex Mitchell. I represent the hero program. Just wanted to let you know that the hero program. Has been in California for the last three and a half years. We're in. Over 20. We're in 28 counties, and we've completed over 29,000 energy efficient projects to make homes. Greener and more energy efficient. And this includes anything. From solar to water. Efficiency. We've done. Almost.\$550 million in home improvements.

#### Compressed Prompt (98 Tokens, 2x Compression):

Item 15 report City Manager Recommendation adopt three resolutions. join Victory Pace program. Second join California first program. three inclusion properties jurisdiction California Hero program. emotion motion second public comment. CNN. cast vote. public comment?. come forward. thank you Mr. Mayor. Members council. Alex Mitchell. represent hero program. California three and a half years. 20. 28 counties completed 29,000 energy efficient projects. includes. solar water. Efficiency Almost \$550 million home improvements.

### Reconstructed prompt (198 Tokens):

Item number 15 on the agenda is the report from the City Manager. The Recommendation is to adopt three resolutions. The first is to join the Victory Pace program. The second resolution is to join the California First program. The third resolution is for the inclusion of properties within our jurisdiction into the California Hero program. There is an emotion for the motion and a second. We will now open the floor for public comment. We have a representative from CNN here. Are there any other individuals who would like to cast a vote or make a public comment? If so, please come forward.

Thank you, Mr. Mayor and Members of the council. My name is Alex Mitchell and I represent the Hero program in California. I have been with the program for three and a half years. As of now, 20 out of 28 counties have completed 29,000 energy efficient projects. This includes projects related to solar water and energy efficiency. Almost \$550 million has been invested in home improvements.

Figure 7: Prompt reconstruction results of our *LLMLingua-2* on the MeetingBank example.

#### <span id="page-12-2"></span>Original Prompt (160 Tokens):

Okay. Thank you. Next step is we're going to do item number, is it that was 16. So I could do item 16. We'll try to get through these as expeditiously as possible. And there's going to be a a motion that's ready to go here. So can we the the the item please. Report from city clerk recommendation to receive and file the certification of the petition regarding the regulation of medical marijuana businesses and approve one of the following three alternative actions adopt the initiative ordinance without alteration to submit the initiative ordinance without alteration to the voters to be held on November 8th, 2016 or three. Adopt a report pursuant to California State Elections Code. Thank you. There's a motion and a second device. Marie Lowenthal. Thank you.

## Compressed Prompt (83 Tokens, 2x compression):

Okay. Thank you. Next step do item number, 16. item 16. try get through expeditiously. motion ready go here. item. Report from city clerk receive file certification of petition regulation medical marijuana businesses approve three alternative actions adopt initiative ordinance without alteration submit to voters November 8th, 2016 or three. Adopt report California State Elections Code. motion second device. Marie Lowenthal.

## Reconstructed Prompt (123 Tokens):

Okay. Thank you. The next step is to do item number 16. For item 16, we will try to get through this expeditiously. The motion is ready to go from here. The item is a report from the city clerk to receive and file the certification of a petition for the regulation of medical marijuana businesses. We can approve three alternative actions: adopt the initiative ordinance without alteration, submit it to the voters on November 8th, 2016, or three, adopt the report as per the California State Elections Code. The motion is seconded by the device. Marie Lowenthal.

Figure 8: Prompt reconstruction results of our *LLMLingua-2* on the MeetingBank example.

tioned filtering process, we retain 3 high-quality question-answer pairs for each meeting transcript. Additionally, we instruct GPT-4-32K to summarize each meeting transcript. The summaries generated by GPT-4 are used as ground truth to evaluate the summary performance.

