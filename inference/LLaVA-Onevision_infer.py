from PIL import Image
import os
import json
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from tqdm import tqdm
from metric import metric

model_id = "lmms-lab/llava-onevision-qwen2-7b-ov"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to("cuda")

processor = AutoProcessor.from_pretrained(model_id)


def gen_response(image_path, q):
    image = Image.open(image_path)
    conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": q},
                    {"type": "image"},
                    ],
                },
            ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors='pt').to("cuda", torch.float16)

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    text = processor.decode(output[0], skip_special_tokens=True)

    start_index = text.find('assistant\n')
    if start_index != -1:
        answer = text[start_index + len('assistant\n'):]
        return answer.lower()
    else:
        print("'assistant\\n' not found")
    
qa_json_path = "Casual-Halbench/qa.json"
dataset_path = "Casual-Halbench/images"
result_path = "llava-onevision-qwen2-7b-qa.json"

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
    
    data.append(info_block)

    # Write back to file
    with open(result_path, 'w') as f:
        json.dump(data, f, indent=4)

metric(qa_json_path, result_path)