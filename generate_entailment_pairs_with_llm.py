import csv
import json
import time
import requests
from typing import List, Tuple


def send_request(
    llm_auth: str,
    model: str,
    query: str
) -> requests.Response:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer " + llm_auth,
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        })
    )
    return response


def load_data(x_path: str, y_path: str) -> List[Tuple[str, str]]:
    with open(x_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]

    with open(y_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f]

    assert len(texts) == len(labels), "Text and label files must be the same length"
    return list(zip(texts, labels))


def generate_entailment_with_openrouter(text: str, llm_auth: str, model: str, max_retries=1) -> str:
    prompt = f"""You are a helpful assistant. Given the following sentence, generate another sentence that is logically entailed by it. The entailed sentence must be true if the original one is true. In your response, do not include any explanations, only return the generated entailment sentence.

Original: "{text}"

Entailed:"""
    for i in range(max_retries):
        try:
            response = send_request(llm_auth, model, prompt)
            if response.status_code == 200:
                result = response.json()
                # print(result)
                content = result["choices"][0]["message"]["content"].strip()
                content = content.lower().strip('"')
                return content
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Exception during entailment generation: {e}")
    
    return None


def create_entailment_pairs(data, output_path, llm_auth, model):
    # Open file and writer once at the start
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sent0', 'sent1'])  # Write header once

        for idx, (text, _) in enumerate(data):
            print(f"[{idx+1}/{len(data)}] Generating entailment...")
            entailment = generate_entailment_with_openrouter(text, llm_auth, model)
            if entailment:
                writer.writerow([text, entailment])
                f.flush()  # Ensure it's written to disk
            else:
                print(f"Skipped due to error for: {text}")
            time.sleep(0.5)  # Avoid rate limiting


def main():
    X_PATH = "data/sarcasm_v2_train.txt"
    Y_PATH = "data/sarcasm_v2_train_label.txt"
    OUTPUT_PATH = "data/sarcasm_v2_deepseek_entailments.csv"

    LLM_AUTH = "LLM_KEY"
    MODEL = "deepseek/deepseek-chat:free"  # or any other supported model like "gpt-3.5-turbo", "mistralai/mistral-7b-instruct", etc.

    data = load_data(X_PATH, Y_PATH)
    create_entailment_pairs(data, OUTPUT_PATH, LLM_AUTH, MODEL)


if __name__ == "__main__":
    main()
