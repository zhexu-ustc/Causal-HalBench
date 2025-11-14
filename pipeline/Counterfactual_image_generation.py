from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm
import time
import numpy as np
import cv2
import torch
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
import json
import os
from openai import OpenAI
import io
import base64

client = genai.Client(api_key="", http_options={'api_version': 'v1alpha'})


def get_objects_by_filename(filename, json_file_path):
    """
    Queries a JSON Lines file (JSONL) to find the 'objects' list for a given image filename.
    """
    # Normalize the filename to ensure it includes the '.jpg' extension for consistent matching
    if not filename.lower().endswith('.jpg'):
        filename_to_match = filename + '.jpg'
    else:
        filename_to_match = filename

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # Check if the 'image' key exists and matches the target filename
                    if 'image' in data and data['image'] == filename_to_match:
                        return data.get('objects') # Return the 'objects' list, or None if key is missing
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON on line {line_num}: {line.strip()}")
                except KeyError as e:
                    print(f"Warning: Missing expected key in JSON on line {line_num}: {e} in {line.strip()}")
        
        # If the loop finishes, the filename was not found
        return None

    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None

def gemini_prompt_gen(prompt_1, prompt_2, image, drawed_image, target_object, counterfactual_object):
    retry = 0
    chat = client.chats.create(model="gemini-2.5-flash",
                                )

    response_1 = chat.send_message([prompt_1, image]).text

    response_2 = chat.send_message([prompt_2, drawed_image])

    response_2 = response_2.text

    return response_2

def prompt_gen(prompt_1, prompt_2, img_base64, drawed_img_base64):
    client = OpenAI(
        api_key="",  
        base_url="",
    )

    messages=[
        {
        "role": "user",
        "content": [                                    
            {
                "type": "text",                        
                "text": prompt_1,                
            },
            {
                "type": "image_url",                
                "image_url": {
                    "url":"data:image/jpeg;base64," + img_base64  
                }
            }
        ]
        
        }
    ]

    response = client.chat.completions.create(
        messages=messages,
        model='gemini-2.5-flash',  
        stream=False
    )

    answer_1 = response.choices[0].message.content

    messages.append({
            "role": "assistant",
            "content": answer_1
        })
    
    messages.append({
            "role": "user",
            "content": [
                {
                "type": "text",                         
                "text": prompt_2,               
            },
            {
                "type": "image_url",                
                "image_url": {
                    "url":"data:image/jpeg;base64," + drawed_img_base64  
                }
            }
            ]
        })
    
    response = client.chat.completions.create(
            model='gemini-2.5-flash',
            messages=messages,
            stream=False
        )
    answer_2 = response.choices[0].message.content

    return answer_2

def draw_mask(img_path,mask_path):
    image = Image.open(img_path).convert("RGBA")  # Original image
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale image

    mask_array = np.array(mask)

    kernel = np.ones((5, 5), np.uint8)  # Define the size of the dilation kernel
    expanded_mask = cv2.dilate(mask_array, kernel, iterations=4) 
    expanded_mask_pil = Image.fromarray(expanded_mask)

    red_layer = Image.new("RGBA", image.size, (255, 0, 0, 80))  # Semi-transparent red (A=80)
    red_overlay = Image.composite(red_layer, Image.new("RGBA", image.size, (0, 0, 0, 0)), expanded_mask_pil)
    result = Image.alpha_composite(image, red_overlay)

    return result

def pil_image_to_base64(pil_image, format='PNG'):
    """
    Convert a PIL.Image.Image object to a pure Base64 encoded string.
    """
    buffered = io.BytesIO()
    # Save the PIL Image object to a memory buffer with specified format
    pil_image.save(buffered, format=format)
    # Get buffer content, encode it in Base64, then decode to UTF-8 string
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_string

PROMPT_1 = "Describe the picture based on the given objects:{list}, be as specific as possible, not too short."
PROMPT_2 = ("Now, if I want to replace only the {target_object} that is specifically colored in red with the {counterfactual_object} using the inpainting model, while leaving any other instances of {target_object} in the image unchanged,"
            "could you give me a concrete and accurate description for the inpainted picture? "
            "Your description should include the {counterfactual_object} and the objects in the just-mentioned object list except for the {target_object}., be as specific as possible, not too short. "
            "And there is no need to mention the replacement operation in the description. "
            "You only need to give me the description without any additional content.")

IMG_PATH = "COCO/val2014/{image_name}.jpg"
MASK_PATH = "COCO/mask/{image_name}/{object}.png"
DILATED_MASK_PATH = "COCO/dilated_mask/{image_name}/{object}.png"

size = (1024, 1024)

# Build pipeline
controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
    )
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)




def Counterfactual_image_generation(save_json, ground_truth_json, output_dir):
    with open(save_json, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    for idx, item in enumerate(results):
        print(f"🔄 Processing item {idx + 1}/{len(results)}: {item['image']},{item['target_object']}->{item['counterfactual_object']}")

        
        image_name = item['image']
        target_object = item['target_object']
        contextual_object = item['contextual_object']
        counterfactual_object = item['counterfactual_object']

        objects_list = get_objects_by_filename(image_name, ground_truth_json)
        formatted_obj_list = str(objects_list).replace('[', '[').replace(']', ']')
        
        # # Load the image and mask
        image_path = IMG_PATH.format(image_name=image_name)
        target_obj_mask_path = MASK_PATH.format(image_name=image_name, object=target_object)
        dilated_target_obj_mask_path = DILATED_MASK_PATH.format(image_name=image_name, object=target_object)

        image = load_image(image_path)
        drawed_image = draw_mask(image_path, target_obj_mask_path)
        image_base64 = pil_image_to_base64(image)
        drawed_image_base64 = pil_image_to_base64(drawed_image)

        prompt_1 = PROMPT_1.format(list=formatted_obj_list)
        prompt_2 = PROMPT_2.format(target_object=target_object, counterfactual_object=counterfactual_object)

        retry = 0

        # prompt generation
        while retry < 6:
            try:
                gemini_prompt = prompt_gen(prompt_1, prompt_2, image_base64, drawed_image_base64)
                break
            except Exception as e:
                print(f"Error: {e}. Retrying...(prompt)")
                retry += 1
                time.sleep(2 ** retry)

        if retry >= 6:
            raise Exception("Max retries exceeded")
        else:
            item["prompt"] = gemini_prompt
            prompt = gemini_prompt
        
        with open(save_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        save_image_path = output_dir + "/" + f"{image_name}-{target_object}_{contextual_object}_{counterfactual_object}.png"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        image = load_image(image_path).convert("RGB").resize(size)
        mask = load_image(dilated_target_obj_mask_path).convert("RGB").resize(size)

        generator = torch.Generator(device="cuda").manual_seed(24)

        # Inpaint
        result = pipe(
            prompt=prompt,
            height=size[1],
            width=size[0],
            control_image=image,
            control_mask=mask,
            num_inference_steps=28,
            generator=generator,
            controlnet_conditioning_scale=0.9,
            guidance_scale=3.5,
            negative_prompt="",
            true_guidance_scale=1.0, # default: 3.5 for alpha and 1.0 for beta
            max_sequence_length=512
        ).images[0]

        result.save(save_image_path)    


