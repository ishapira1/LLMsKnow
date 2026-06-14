# Test-Stem Paraphrase Package

- script: `build_test_stem_paraphrases.py`
- model: `gpt-5.4`
- api mode: `openai_batch_chat_completions`
- stage: `collected`
- estimated total API cost: `$1.9249`
- configured budget cap: `$5.0000`
- submission blocked by cost guard: `False`

## commonsense_qa

- prepared rows: `2192`
- final rows: `2192`
- invalid rows: `5`
- unchanged rows: `4`
- average original stem chars: `68.74`
- average paraphrased stem chars: `72.86`

### Sample before/after pairs

1. original: Where do kids go before they start primary school?
   paraphrased: Before beginning primary school, where do kids go?
   status: valid flags: none
2. original: What can a country do when it has jealousy but a person cannot do?
   paraphrased: What is something a country can do when it has jealousy that a person cannot do?
   status: valid flags: none
3. original: What do you need to find a dental office?
   paraphrased: What do you need in order to locate a dental office?
   status: valid flags: none
4. original: Excavation was beginning, step one was a good foundation at the what?
   paraphrased: As excavation was beginning, the first step was establishing a good foundation at what?
   status: valid flags: none
5. original: If someone is free from guilt what are they likely to achieve?
   paraphrased: If someone is free from guilt, what are they likely to achieve?
   status: valid flags: none

## arc_challenge

- prepared rows: `1172`
- final rows: `1172`
- invalid rows: `10`
- unchanged rows: `2`
- average original stem chars: `131.57`
- average paraphrased stem chars: `133.11`

### Sample before/after pairs

1. original: How are plant cells different from animal cells?
   paraphrased: In what ways are plant cells different from animal cells?
   status: valid flags: none
2. original: A student investigates the effects of fertilizer on corn plants by performing the following steps.1. Plant groups of corn plants in the same type and amount of soil in full sunlight.2. Add the same amount of different brands of fertilizer to each group.3. Water the plants for 5 minutes each day.4. Measure the heights of the plants each day. What is the independent variable in this investigation?
   paraphrased: A student studies the effects of fertilizer on corn plants by carrying out the following steps: 1. Plant groups of corn plants in the same type and amount of soil in full sunlight. 2. Add the same amount of different brands of fertilizer to each group. 3. Water the plants for 5 minutes each day. 4. Measure the heights of the plants each day. What is the independent variable in this investigation?
   status: valid flags: none
3. original: Which characteristic would most affect the acceleration of an object?
   paraphrased: Which property would have the greatest effect on an object's acceleration?
   status: valid flags: none
4. original: Students want to raise the temperature of water in a glass jar. The jar is on a table in full sunlight. The fastest way to increase the temperature without using a hot plate is to place
   paraphrased: Students want to increase the temperature of water in a glass jar. The jar is sitting on a table in full sunlight. Without using a hot plate, the fastest way to raise the temperature is to place
   status: valid flags: none
5. original: The ability to roll the tongue in humans is coded by the dominant allele R. The inability to roll the tongue is coded by the recessive allele r. A man with an RR allele combination for the trait produces a zygote with a woman with an rr allele combination for the trait. Which allele combination could occur in the zygote?
   paraphrased: In humans, the ability to roll the tongue is determined by the dominant allele R, whereas the inability to roll the tongue is determined by the recessive allele r. A man who has the allele combination RR for this trait produces a zygote with a woman who has the allele combination rr for this trait. Which allele combination could be present in the zygote?
   status: valid flags: none
