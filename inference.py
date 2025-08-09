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

    prompt_parts = [
        'Context:',
        f'- id: {sample_id}',
        f'- image_id: {image_id}',
        f'- question: {question}',
        '',
        'Task: Based on the image and your knowledge of traffic laws, answer the question.',
        'Return ONLY a JSON object in this exact schema (no additional text, no explanations):',
        '{',
        '  "id": "<copy the id above>",',
        '  "image_id": "<copy the image_id above>",',
        '  "question": "<copy the question above>",',
        '  "question_type": "<copy the question_type above>",',
        '  "choices": {},',
        answer_schema_line,
        '  "relevant_articles": [',
        '    {"law_id": "<law or standard id>", "article_id": "<article number>"}',
        '  ]',
        '}',
        'Rules:',
        '- Respond in Vietnamese only.',
        '- Output valid JSON only (no markdown, no prose).',
        '- Use keys exactly: id, image_id, question, answer, relevant_articles.',
        '- Do not output placeholders like <...>; copy actual values from Context.',
        '- Use double quotes for all keys and string values; no trailing commas.',
        '- For each item in relevant_articles, set law_id to the law object\'s "id" and article_id to the nested article object\'s "id" (from the law database).',
        '- Valid law_id values: "QCVN 41:2024/BGTVT" and "36/2024/QH15" only. Do NOT invent other law names.',
        '- article_id must be a string numeric identifier like "22" (not "Article 22")',
        '- Include "question_type" and "choices" in the JSON. If Multiple choice, set choices to an object with keys A,B,C,D copying texts exactly; otherwise set choices to {}.',
        '- For Multiple choice, answer must be exactly one of: "A", "B", "C", or "D". For Yes/No, answer must be exactly "Đúng" or "Sai".',
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

    # Extract and return the first JSON object from the decoded text without modification
    json_match = re.search(r"\{[\s\S]*\}", decoded_text)
    if json_match:
        return json_match.group(0).strip()
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
        

        # Find the first occurrence of "json" (case-insensitive) in answer_json_str, if any
        json_pos = answer_json_str.lower().find("```json")
        if json_pos != -1:
            print(f'Found "json" at position {json_pos} in answer_json_str.')
        answer_json_str = answer_json_str[json_pos + 4:]

        print("answer_json_str: ```````", answer_json_str, "``````")
    
        try:
            answer_data = json.loads(answer_json_str)
        except json.JSONDecodeError:
            answer_data = {"answer": "Error decoding JSON", "relevant_articles": []}

        results.append({
            "id": sample["id"],
            "image_id": image_id,
            "question": question,
            "answer": answer_data.get("answer", ""),
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