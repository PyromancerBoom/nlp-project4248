import ollama
import pandas as pd
import time
from tqdm import tqdm
import csv
import pickle

with open('sarcasm_classifications.csv', encoding="utf-8") as f:
    csv_reader = csv.DictReader(f)
    results = [row for row in csv_reader]

with open('new_sarcasm_classifications.csv', encoding="utf-8") as f:
    csv_reader = csv.DictReader(f)
    new_results = [row for row in csv_reader]

changed_indices = []

for i in range(len(results)):
    h, c = results[i].values()
    nc = new_results[i]['classification']
    if c != nc:
        changed_indices.append(i)    

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
definitions = {
    "contextual contradiction": "Sarcasm that presents a situation that contradicts common knowledge, expectations, or reality.",
    "mock enthusiasm": "Sarcasm that expresses false excitement or positive sentiment about clearly negative situations.",
    "stylistic irony": "Sarcasm conveyed through specific writing techniques like quotation marks, exaggeration, or structural patterns.", 
    "institutional critique": "Sarcasm that mocks institutions, systems, or authority figures by highlighting absurd or hypocritical behaviors.", 
    "behavioral observation": "Sarcasm that highlights common human behaviors or social patterns in an exaggerated or formalized way."
}

prompt_template_3 = """
You are an expert in sarcasm classification. Two previous analyses of this headline gave different classifications:

Headline: "{headline}"
Classification 1: "{original_classification}"
Classification 2: "{suggested_classification}"

Here are the definitions of both categories:
- {original_classification}: {original_class_definition}
- {suggested_classification}: {suggested_class_definition}

Carefully evaluate which classification better captures the headline's primary form of sarcasm. Consider specific words, patterns, and contextual elements in the headline.

Respond with only the category name that best fits the headline.
"""

prompt_template = ''.join([system_prompt, prompt_template_3])

for i in tqdm(changed_indices, desc="Finalizing classifications", unit="headline"):
    headline, classification_a = results[i]["headline"], results[i]["classification"]
    classification_b = new_results[i]["classification"]
    try:
        response = ollama.chat(
            model="llama3:8b", 
            messages=[{"role": "user", "content": prompt_template.format(
                headline=headline, 
                original_classification=classification_a, 
                suggested_classification=classification_b,
                original_class_definition=definitions[classification_a],
                suggested_class_definition=definitions[classification_b])}]
        )
        classification = response['message']['content'].strip()
        classification = classification.lower()


        for subtype in ["contextual contradiction", "mock enthusiasm", "stylistic irony", "institutional critique", "behavioral observation"]:
            if subtype in classification:
                classification = subtype
                break
        
        results[i]["classification"] = classification
        time.sleep(0.1)
    except Exception as e:
        print(f"Error processing headline: {headline[:30]}... - {str(e)}")
        time.sleep(1)

pd.DataFrame(results).to_csv('final_sarcasm_classifications.csv', index=False)