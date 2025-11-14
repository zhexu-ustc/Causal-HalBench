from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import os
import json
from tqdm import tqdm
from metric import metric

model_path = "moonshotai/Kimi-VL-A3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

qa_json_path = "Casual-Halbench/qa.json"
dataset_path = "Casual-Halbench/images"
result_path = "Kimi-VL-A3B-Instruct-qa.json"


def gen_response(image_path, prompt):
    image = Image.open(image_path)
    messages = [
    {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return answer

if not os.path.exists(result_path):
    with open(result_path, 'w') as f:
        json.dump([], f)
    print(f"File {result_path} does not exist, created and initialized as empty list")
else:
    print(f"File {result_path} already exists")


with open(qa_json_path, 'r') as f:
    qa_data = json.load(f)  

for item in tqdm(qa_data):
    image_name = item['image_name']
    q = item['q']
    type = item['type']
    tag = item['tag']
    id = item['id']

    image_path = os.path.join(dataset_path, image_name)
    answer = gen_response(image_path, q)

    if "yes" in answer.lower():
        answer = "yes"
    else:
        answer = "no"
    info_block = {
                "image_name": image_name,
                "type": type,
                "answer": answer,
                "tag": tag,
                "id": id
            }
    
    with open(result_path, 'r') as f:
                data = json.load(f)

# Append new data
    data.append(info_block)

    # Write back to file
    with open(result_path, 'w') as f:
        json.dump(data, f, indent=4)

metric(qa_json_path, result_path)

