# import json
# import os
# import re
# from datasets import Dataset, Image

# def _load_law_index(law_db_path: str):
#     """Load law DB and index by (law_id, article_id) -> {title, text}."""
#     if not os.path.exists(law_db_path):
#         return {}
#     try:
#         with open(law_db_path, "r") as f:
#             law_db = json.load(f)
#     except Exception:
#         return {}

#     law_index = {}
#     # law_db is expected to be a list of laws, each with id and articles
#     for law in law_db:
#         law_id = law.get("id")
#         for article in law.get("articles", []):
#             article_id = str(article.get("id"))
#             raw_text = article.get("text", "")
#             # Replace embedded IMAGE and TABLE placeholders to keep prompt clean
#             # Capture image filename
#             text = re.sub(r"<<IMAGE:\s*(.*?)\s*/IMAGE>>", r"[IMAGE: \\1]", raw_text)
#             # Omit large tables
#             text = re.sub(r"<<TABLE:[\s\S]*?/TABLE>>", "[TABLE omitted]", text)
#             # Normalize whitespace a bit
#             text = re.sub(r"\n{3,}", "\n\n", text).strip()
#             law_index[(law_id, article_id)] = {
#                 "title": article.get("title", ""),
#                 "text": text,
#             }
#     return law_index


# def _excerpt(text: str, max_chars: int = 600) -> str:
#     if len(text) <= max_chars:
#         return text
#     return text[: max_chars - 3].rstrip() + "..."


# def create_dataset(json_path, image_dir, law_db_path: str = "dataset/db/vlsp2025_law.json"):
#     law_index = _load_law_index(law_db_path)
#     with open(json_path, "r") as f:
#         train_data = json.load(f)

#     processed_data = {
#         "image": [],
#         "prompt": [],
#         "output": [],
#     }

#     for sample in train_data:
#         image_path = os.path.join(image_dir, f"{sample['image_id']}.jpg")
#         if not os.path.exists(image_path):
#             image_path = os.path.join(image_dir, f"{sample['image_id']}.png")
#             if not os.path.exists(image_path):
#                 continue

#         # Build optional law context from DB based on relevant_articles
#         law_context_lines = []
#         # Limit number of law excerpts to keep prompt length under control
#         for ra in sample.get("relevant_articles", [])[:2]:
#             law_id = ra.get("law_id")
#             article_id = str(ra.get("article_id"))
#             meta = law_index.get((law_id, article_id))
#             if not meta:
#                 continue
#             title = meta.get("title", "").strip()
#             text_excerpt = _excerpt(meta.get("text", ""))
#             header = f"- {law_id} - Article {article_id}"
#             if title:
#                 header += f": {title}"
#             law_context_lines.append(header + "\n" + text_excerpt)
#         law_context = "\n".join(law_context_lines)

#         prompt_parts = [
#             "Context:",
#             f"- id: {sample['id']}",
#             f"- image_id: {sample['image_id']}",
#             f"- question: {sample['question']}",
#             "",
#         ]
#         if law_context:
#             prompt_parts += [
#                 "Law database excerpts:",
#                 law_context,
#                 "",
#             ]
#         prompt_parts += [
#             'Task: Based on the image and your knowledge of traffic laws, and the excerpts above, answer the question.',
#             'Return ONLY a JSON object in this exact schema (no additional text, no explanations):',
#             '{',
#             '  "id": "<copy the id above>",',
#             '  "image_id": "<copy the image_id above>",',
#             '  "question": "<copy the question above>",',
#             '  "answer": "<concise answer in English>",',
#             '  "relevant_articles": [',
#             '    {"law_id": "<law or standard id>", "article_id": "<article number>"}',
#             '  ]',
#             '}',
#             'Rules:',
#             '- Respond in English only.',
#             '- Output valid JSON only (no markdown, no prose).',
#             '- Use keys exactly: id, image_id, question, answer, relevant_articles.',
#             '- Do not output placeholders like <...>; copy actual values from Context.',
#             '- Use double quotes for all keys and string values; no trailing commas.',
#             '- If no law is clearly relevant, return an empty array for relevant_articles.',
#             '- For each item in relevant_articles, set law_id to the law object\'s "id" and article_id to the nested article object\'s "id" (from the law database).',
#             '- Valid law_id values: "QCVN 41:2024/BGTVT" and "36/2024/QH15" only. Do NOT invent other law names.',
#             '- article_id must be a string numeric identifier like "22" (not "Article 22"). If uncertain, use an empty array for relevant_articles.',
#         ]
#         prompt = "\n".join(prompt_parts)
#         if sample.get('choices'):
#             choices = "\n".join([f"{key}: {value}" for key, value in sample['choices'].items()])
#             prompt += f"\n{choices}"

#         output_json = json.dumps({
#             "id": sample["id"],
#             "image_id": sample["image_id"],
#             "question": sample["question"],
#             "answer": sample["answer"],
#             "relevant_articles": sample["relevant_articles"]
#         }, ensure_ascii=False)

#         processed_data["image"].append(image_path)
#         processed_data["prompt"].append(prompt)
#         processed_data["output"].append(output_json)

#     return Dataset.from_dict(processed_data).cast_column("image", Image())

# def convert_to_conversation(sample):
#     return {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": sample["image"]},
#                     {"type": "text", "text": sample["prompt"]}
#                 ]
#             },
#             {
#                 "role": "assistant",
#                 "content": [
#                     {"type": "text", "text": sample["output"]}
#                 ]
#             }
#         ]
#     }

# if __name__ == '__main__':
#     json_path = "dataset/train/vlsp_2025_train.json"
#     image_dir = "dataset/train/train_images"

#     structured_dataset = create_dataset(json_path, image_dir)
#     converted_dataset = [convert_to_conversation(sample) for sample in structured_dataset]

#     print(f"Successfully processed {len(converted_dataset)} samples.")
#     if converted_dataset:
#         print("First sample:")
#         print(json.dumps(converted_dataset[0], ensure_ascii=False, indent=2))

import json
import os
import re
from datasets import Dataset, Image

def _load_law_index(law_db_path: str):
    """Load law DB and index by (law_id, article_id) -> {title, text}."""
    if not os.path.exists(law_db_path):
        return {}
    try:
        with open(law_db_path, "r") as f:
            law_db = json.load(f)
    except Exception:
        return {}

    law_index = {}
    # law_db is expected to be a list of laws, each with id and articles
    for law in law_db:
        law_id = law.get("id")
        for article in law.get("articles", []):
            article_id = str(article.get("id"))
            raw_text = article.get("text", "")
            # Extract image filename(s) and remove IMAGE/TABLE blocks from text
            image_matches = re.findall(r"<<IMAGE:\s*(.*?)\s*/IMAGE>>", raw_text)
            for image_match in image_matches:
                image_filename = image_match
                text_wo_images = re.sub(r"<<IMAGE:\s*(.*?)\s*/IMAGE>>", "", raw_text)
                text_wo_tables = re.sub(r"<<TABLE:\s*(.*?)\s*/TABLE>>", "", text_wo_images)
                # Normalize whitespace
                cleaned_text = re.sub(r"\n{3,}", "\n\n", text_wo_tables).strip()
                if (law_id, article_id) not in law_index:
                    law_index[(law_id, article_id)] = []
                law_index[(law_id, article_id)].append({
                    "article_id": article_id,
                    "law_id": law_id,
                    "article_title": article.get("title", ""),
                    "text": cleaned_text,
                    "image": image_filename,
                })
    return law_index


def _excerpt(text: str, max_chars: int = 600) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def create_dataset(json_path, image_dir, law_db_path: str = "dataset/db/vlsp2025_law.json"):
    law_data = _load_law_index(law_db_path)
    with open(json_path, "r") as f:
        train_data = json.load(f)

    conversations = []
    
    # Add law article understanding tasks (multi-task learning)
    for law in law_data.values():
        for law_item in law:
            if not law_item.get("image"):
                continue
            user_content = [
                {"type": "image", "image": f"dataset/db/images.fld/{law_item['image']}"},
                {
                    "type": "text",
                    "text": f"Explain this traffic law sign according to {law_item['law_id']} - Article {law_item['article_id']}"
                    + (f": {law_item['article_title']}" if law_item.get("article_title") else "")
                },
            ]
            conversations.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": law_item["text"]}]},
                ]
            })

    # Add question-answer tasks with law context
    for sample in train_data:
        image_id = sample.get("image_id")
        question = sample.get("question")
        answer = sample.get("answer")
        relevant_articles = sample.get("relevant_articles", [])
        question_type = sample.get("question_type", "")
        choices = sample.get("choices")

        # Resolve image path
        img_path_jpg = os.path.join("dataset/train/train_images", f"{image_id}.jpg")
        img_path_png = os.path.join("dataset/train/train_images", f"{image_id}.png")
        image_path = img_path_jpg if os.path.exists(img_path_jpg) else (img_path_png if os.path.exists(img_path_png) else None)
        if image_path is None:
            continue

        # Build law context from relevant articles
        law_context_lines = []
        for ra in relevant_articles[:2]:  # Limit to prevent prompt overflow
            law_id = ra.get("law_id")
            article_id = str(ra.get("article_id"))
            law_items = law_data.get((law_id, article_id), [])
            for law_item in law_items[:1]:  # Take first match
                title = law_item.get("article_title", "").strip()
                text_excerpt = _excerpt(law_item.get("text", ""))
                header = f"- {law_id} - Article {article_id}"
                if title:
                    header += f": {title}"
                law_context_lines.append(header + "\n" + text_excerpt)

        # Build enhanced prompt with law context
        prompt_parts = [
            f"Question: {question}",
        ]
        if question_type:
            prompt_parts.append(f"Question type: {question_type}")
        if choices:
            rendered = ", ".join([f"{k}: {v}" for k, v in choices.items()])
            prompt_parts.append(f"Choices: {rendered}")
        
        if law_context_lines:
            prompt_parts.extend([
                "",
                "Relevant law excerpts:",
                "\n".join(law_context_lines),
                "",
                "Based on the image and law excerpts above, answer the question."
            ])
        else:
            prompt_parts.append("Based on the image and your knowledge of traffic laws, answer the question.")

        user_content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "\n".join(prompt_parts)},
        ]

        # Enhanced assistant response with JSON format
        output_json = json.dumps({
            "id": sample.get("id"),
            "image_id": image_id,
            "question": question,
            "answer": answer,
            "relevant_articles": relevant_articles
        }, ensure_ascii=False)

        conversations.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": output_json}]},
            ]
        })

    return conversations

if __name__ == '__main__':
    json_path = "dataset/train/vlsp_2025_train.json"
    image_dir = "dataset/train/train_images"

    messages = create_dataset(json_path, image_dir)

    print(f"Successfully processed {len(messages)} samples.")
    if messages:
        print("First sample:")
        print(json.dumps(messages[0], ensure_ascii=False, indent=2))