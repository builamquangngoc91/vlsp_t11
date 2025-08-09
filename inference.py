import torch
from unsloth import FastVisionModel
from transformers import TextStreamer
from PIL import Image
import json
import re
import os
from tqdm import tqdm

def run_inference(model, tokenizer, sample_id, image_id, image_path, question, question_type=None, choices=None):
    """
    Runs inference on a single image and question, returning the generated JSON string.
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Warning: Image not found at {image_path}. Skipping.")
        return "{}"

    
    # Prepare the prompt
    # Base prompt
    answer_schema_line = '  "answer": "<concise answer in Vietnamese>",' \
        if question_type not in ("Multiple choice", "Yes/No") else \
        ('  "answer": "<A|B|C|D>",' if question_type == "Multiple choice" else '  "answer": "<Đúng|Sai>",')

    # Build the JSON template with actual values (avoid placeholders)
    prompt_parts = [
        'Context:',
        f'- id: {sample_id}',
        f'- image_id: {image_id}',
        f'- question: {question}',
        '',
        'Task: Based on the image and your knowledge of traffic laws, answer the question.',
        'Return ONLY a JSON object in this exact schema (no additional text, no explanations):',
        '{',
        f'  "id": "{sample_id}",',
        f'  "image_id": "{image_id}",',
        f'  "question": "{question}",',
    ]

    # If Task 2 (question_type provided), include question_type/choices/answer; Task 1 omits them
    if question_type:
        prompt_parts.append(f'  "question_type": "{question_type}",')
        if question_type == "Multiple choice" and isinstance(choices, dict) and choices:
            ordered_keys = [k for k in ["A", "B", "C", "D"] if k in choices]
            choices_items = ", ".join([f'"{k}": "{choices[k]}"' for k in ordered_keys])
            prompt_parts.append(f'  "choices": {{{choices_items}}},')
            prompt_parts.append('  "answer": "<A|B|C|D>",')
        elif question_type == "Yes/No":
            prompt_parts.append('  "choices": {},')
            prompt_parts.append('  "answer": "<Đúng|Sai>",')
        else:
            prompt_parts.append('  "choices": {},')
            prompt_parts.append(answer_schema_line)

    # Common tail for both tasks
    prompt_parts += [
        '  "relevant_articles": [',
        '    {"law_id": "<law or standard id>", "article_id": "<article number>"}',
        '  ]',
        '}',
        'Rules:',
        '- Respond in Vietnamese.',
        '- Keep "question" and "choices" in Vietnamese; do not translate or paraphrase.',
        '- Output valid JSON only (no markdown, no prose).',
        '- For Task 1 (no question_type): use keys exactly: id, image_id, question, relevant_articles.',
        '- For Task 2 (has question_type): use keys exactly: id, image_id, question, question_type, choices, answer, relevant_articles.',
        '- Do not output placeholders like <...>; copy actual values from Context.',
        '- Use double quotes for all keys and string values; no trailing commas.',
        '- For each item in relevant_articles, set law_id to the law object\'s "id" and article_id to the nested article object\'s "id" (from the law database).',
        '- Valid law_id values: "QCVN 41:2024/BGTVT" and "36/2024/QH15" only. Do NOT invent other law names.',
        '- article_id must be a string numeric identifier like "22" (not "Article 22")',
    ]

    # Inject question_type and choices guidance when available
    if question_type:
        prompt_parts.append(f"- question_type: {question_type}")
        if question_type == "Multiple choice" and choices:
            rendered = "\n".join([f"{k}: {v}" for k, v in choices.items()])
            prompt_parts += [
                "- Choices:",
                rendered,
                "- For Multiple choice, answer must be exactly one of: \"A\", \"B\", \"C\", or \"D\" (no explanation).",
            ]
        elif question_type == "Yes/No":
            prompt_parts += [
                "- For Yes/No, answer must be exactly one of: \"Đúng\" or \"Sai\" (no explanation).",
            ]

    prompt = "\n".join(prompt_parts)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
    ).to("cuda")

    # Generate the response
    generated_ids = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract and return the first balanced JSON object from the decoded text
    def _first_json_obj(text: str):
        start = text.find('{')
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:i+1]
                            try:
                                json.loads(candidate)
                                return candidate
                            except json.JSONDecodeError:
                                break
            start = text.find('{', start + 1)
        return None

    json_str = _first_json_obj(decoded_text)
    if json_str:
        return json_str.strip()
    return "{}"

def main():
    print("Loading fine-tuned model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="lora_model",
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    print("Model loaded successfully.")

    test_json_path = "dataset/test/vlsp_2025_public_test_task1.json"
    image_dir = "dataset/test/public_test_images"
    with open(test_json_path, "r") as f:
        test_data = json.load(f)

    results = []
    
    print(f"Running inference on {len(test_data)} test samples...")
    for sample in tqdm(test_data):
        image_id = sample["image_id"]
        question = sample["question"]
        
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(image_dir, f"{image_id}.png")

        answer_json_str = run_inference(
            model,
            tokenizer,
            sample["id"],
            image_id,
            image_path,
            question,
            sample.get("question_type"),
            sample.get("choices"),
        )

        print("answer_json_str: ```````", answer_json_str, "``````")

        # Remove any text before the first '{' and after the last '}' in answer_json_str (keep only the JSON object)
        # Also, if "json" appears in the string, remove everything before and including "json"
        answer_json_str = answer_json_str.strip()
        json_pos = answer_json_str.lower().find("json")
        if json_pos != -1:
            # Remove everything before and including "json"
            answer_json_str = answer_json_str[json_pos + len("json") :]
            # Now, find the first '{' after "json"
            first_brace = answer_json_str.find('{')
            last_brace = answer_json_str.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                answer_json_str = answer_json_str[first_brace:last_brace+1]
            else:
                answer_json_str = "{}"
        else:
            # No "json" in string, just extract the first JSON object
            first_brace = answer_json_str.find('{')
            last_brace = answer_json_str.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                answer_json_str = answer_json_str[first_brace:last_brace+1]
            else:
                answer_json_str = "{}"

        
        try:
            
            answer_data = json.loads(answer_json_str)
        except json.JSONDecodeError:
            answer_data = {"answer": "Error decoding JSON", "relevant_articles": []}

        results.append({
            "id": sample["id"],
            "image_id": image_id,
            "question": question,
            "relevant_articles": answer_data.get("relevant_articles", [])
        })
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer_data.get('answer', 'N/A')}")
        print("-" * 20)

    output_file = "test_results_task1.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nInference complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
