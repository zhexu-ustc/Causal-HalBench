import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import os
import json
from tqdm import tqdm
from metric import metric

# load omni model default, the default init_vision/init_audio/init_tts is True
# if load vision-only model, please set init_audio=False and init_tts=False
# if load audio-only model, please set init_vision=False
model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-o-2_6',
    trust_remote_code=True,
    attn_implementation='sdpa', # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=True,
    init_tts=True
)


model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)


def gen_response(image_path, q):
    image = Image.open(image_path).convert('RGB')

    msgs = [{'role': 'user', 'content': [image, q]}]
    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer
    )
    return answer


qa_json_path = "Casual-Halbench/qa.json"
dataset_path = "Casual-Halbench/images"
result_path = "MiniCPM-o-2_6-qa.json"


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
