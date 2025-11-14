from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
from tqdm import tqdm
import os
import json
from metric import metric

processor = LlavaNextProcessor.from_pretrained("lmms-lab/llama3-llava-next-8b")

model = LlavaNextForConditionalGeneration.from_pretrained("lmms-lab/llama3-llava-next-8b", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")

qa_json_path = "Casual-Halbench/qa.json"
dataset_path = "Casual-Halbench/images"
result_path = "llama3-llava-next-8b-qa.json"



def gen_response(image_path,prompt):
    image = Image.open(image_path).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)

    text = processor.decode(output[0], skip_special_tokens=True)

    start_index = text.find("assistant\n\n\n")
    if start_index != -1:
        result = text[start_index + len("assistant\n\n\n"):]
    return result



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

    answer = gen_response(image_path, q).lower()

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
