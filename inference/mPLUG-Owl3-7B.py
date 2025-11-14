import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
import os
import json
from tqdm import tqdm
from metric import metric


model_path = 'mPLUG/mPLUG-Owl3-7B-241101'
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half, trust_remote_code=True)
model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)


qa_json_path = "Casual-Halbench/qa.json"
dataset_path = "Casual-Halbench/images"
result_path = "mPLUG-Owl3-7B-qa.json"


def gen_response(image_path, q):
    image = Image.open(image_path).convert("RGB")

    messages = [
                {"role": "user", "content": f"<|image|>\n{q}"},
                {"role": "assistant", "content": ""}
            ]
    
    inputs = processor(messages, images=[image], videos=None)
    inputs.to('cuda')
    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens':100,
        'decode_text':True,
    })

    answer = model.generate(**inputs)[0].lower()
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


