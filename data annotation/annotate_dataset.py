import ollama
import pandas as pd
import time
from tqdm import tqdm

df = pd.read_json('news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json', lines=True)

sarcastic_headlines = df['headline'][df['is_sarcastic'] == 1].to_list()


system_prompt = """
You are an expert in linguistic analysis specializing in sarcasm detection. Your task is to classify sarcastic headlines into one of these five distinct categories. Here are examples of each category:

1. CONTEXTUAL CONTRADICTION: Sarcasm that presents a situation that contradicts common knowledge, expectations, or reality.
Example: "Study finds regular exercise better than sitting on couch all day"
Example: "Local man shocked to discover fast food isn't healthy"

2. MOCK ENTHUSIASM: Sarcasm that expresses false excitement or positive sentiment about clearly negative situations.
Example: "Can't wait to spend my weekend fixing the plumbing!"
Example: "Awesome! 3-hour flight delay with screaming toddler nearby"

3. STYLISTIC IRONY: Sarcasm conveyed through specific writing techniques like quotation marks, exaggeration, or structural patterns.
Example: "New diet book promises to help readers lose 'up to' zero pounds"
Example: "Area man writes absolutely perfect, flawless email with zeroooo mistakes"

4. INSTITUTIONAL CRITIQUE: Sarcasm that mocks institutions, systems, or authority figures by highlighting absurd or hypocritical behaviors.
Example: "Congress introduces bill to help struggling Americans by losing their tax returns"
Example: "Health insurer celebrates record profits by denying more claims"

5. BEHAVIORAL OBSERVATION: Sarcasm that highlights common human behaviors or social patterns in an exaggerated or formalized way.
Example: "Dad enters 15th consecutive hour of watching 'just five minutes' of golf"
Example: "Man completely transforms personality while talking to waiter"

Here are some headlines I've already classified correctly:
- "Area Man Overjoyed To Receive Seventh Credit Card Offer This Week" = MOCK ENTHUSIASM
- "Rick Moranis to star in straight-to-video release honey, i shrunk some more shit" = STYLISTIC IRONY
- "Depressed groundhog sees shadow of rodent he once was" = STYLISTIC IRONY
- "Youtuber wastes 2 whole minutes explaining how to prep a deck for sealant as if viewer total moron" = BEHAVIORAL OBSERVATION
- "Nader supporters blame electoral defeat on bush, kerry" = INSTITUTIONAL CRITIQUE
- "Family mercifully pulling plug on grandfather unaware they sending him directly to hell" = CONTEXTUAL CONTRADICTION

"""

prompt_template_1 = """
Your current task is to classify sarcastic headlines into one of these five distinct categories.

Headline to classify: "{headline}"

Think step by step about what makes this headline sarcastic. Which of the five categories (CONTEXTUAL CONTRADICTION, MOCK ENTHUSIASM, STYLISTIC IRONY, INSTITUTIONAL CRITIQUE, or BEHAVIORAL OBSERVATION) best describes the PRIMARY type of sarcasm?

Answer with only the category name.
"""
prompt_template = ''.join([system_prompt, prompt_template_1])

results = []
for i in tqdm(range(len(sarcastic_headlines)), desc="Classifying headlines", unit="headline"):
    headline = sarcastic_headlines[i]
    try:
        response = ollama.chat(
            model="llama3:8b", 
            messages=[{"role": "user", "content": prompt_template.format(headline=headline)}]
        )
        classification = response['message']['content'].strip()
        classification = classification.lower()

        for subtype in ["contextual contradiction", "mock enthusiasm", "stylistic irony", "institutional critique", "behavioral observation"]:
            if subtype in classification:
                classification = subtype
                break
        
        results.append({"headline": headline, "classification": classification})
        time.sleep(0.1)
    except Exception as e:
        print(f"Error processing headline: {headline[:30]}... - {str(e)}")
        time.sleep(1)
    
    # Save results every 100 headlines
    if len(results) % 100 == 0:
        pd.DataFrame(results).to_csv(f'sarcasm_classifications.csv', index=False)

pd.DataFrame(results).to_csv('sarcasm_classifications.csv', index=False)

print("Initial labeling completed!")