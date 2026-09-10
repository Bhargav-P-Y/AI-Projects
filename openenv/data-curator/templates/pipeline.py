import json
from filter_logic import is_safe

INPUT_FILE = "raw_dataset.jsonl"
OUTPUT_FILE = "processed_dataset.jsonl"

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_llama3(row):
    """
    BUG: Currently outputs generic text. Needs to be formatted with strict Llama 3 tokens:
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{completion}<|eot_id|>
    """
    prompt = row.get('prompt', '')
    completion = row.get('completion', '')

    # Needs to be replaced!
    formatted_text = f"User: {prompt}\nAssistant: {completion}"

    return {"text": formatted_text}

def scrub_pii(row):
    """
    BUG: Currently does nothing. Needs to replace phone numbers (e.g., 555-0198) with [REDACTED].
    """
    text = row.get('text', '')
    # TODO: Implement regex to scrub phone numbers
    row['text'] = text
    return row

def process_pipeline():
    print("Starting Data Curation Pipeline...")
    raw_data = load_data(INPUT_FILE)

    processed_data = []
    for row in raw_data:
        # 1. Format
        formatted_row = format_llama3(row)

        # 2. Scrub PII
        scrubbed_row = scrub_pii(formatted_row)

        # 3. Filter Toxicity
        if is_safe(scrubbed_row['text']):
            processed_data.append(scrubbed_row)

    # BUG: No deduplication step exists! The agent must add logic to drop duplicate texts before saving.

    # Save output
    with open(OUTPUT_FILE, 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")

    print(f"Pipeline complete. Saved {len(processed_data)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_pipeline()
