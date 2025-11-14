import json
from google import genai
from PIL import Image
import random
import time
import numpy as np
import cv2
from openai import OpenAI
import io
import base64
import os


client = genai.Client(api_key="", http_options={'api_version': 'v1alpha'})

def get_objects_by_filename(filename, json_file_path):
    """
    Queries a JSON Lines file (JSONL) to find the 'objects' list for a given image filename.
    """
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

def send_message(question, img_base64):
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
                "text": question,                
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

    answer = response.choices[0].message.content

    return answer

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

Q_TEMPLATE = ("I now want to focus specifically on the {target_object} that is colored in red in the image, and replace only this specific {target_object} with another object."
            "Here are the replacement items to choose from: {list} "
            "Which item do you think would be more suitable in terms of size and how well it fits the image? "
            "Please output only one item and do not include any other output.")
IMG_PATH = "COCO/val2014/{image_name}.jpg"
MASK_PATH = "COCO/added_mask_1/{image_name}/{target_object}.png"


obj_list = [ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]


def Intervention_objects_selection(save_json, ori_img_txt, co_count_json, ground_truth_json):
    """
    Main function to process images, find objects, and generate replacement candidates.
    """
    print(" Starting the process...")
    # 1. Load and preprocess co-occurrence data
    print(f"Loading co-occurrence data from '{co_count_json}'...")
    try:
        with open(co_count_json, 'r', encoding='utf-8') as f:
            co_occurrence_data = json.load(f)
    except FileNotFoundError:
        print(f" Error: Co-occurrence JSON file not found.")
        return
    except json.JSONDecodeError:
        print(f" Error: Could not decode JSON from the co-occurrence file.")
        return

    # Convert keys in "Co-occurs with person" format to "person"
    def parse_co_occurrence_dict(raw_dict):
        return {key.replace("Co-occurs with ", ""): value for key, value in raw_dict.items()}

    processed_co_data = {obj: parse_co_occurrence_dict(data) for obj, data in co_occurrence_data.items()}
    all_possible_objects = set(obj_list)
    with open(save_json, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 2. Read list of original image names
    print(f"Reading image filenames from '{ori_img_txt}'...")
    try:
        with open(ori_img_txt, 'r', encoding='utf-8') as f:
            image_filenames = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f" Error: Image list file not found at '{ori_img_txt}'")
        return

    # 3. Process each image
    for idx, filename in enumerate(image_filenames):
        print(f"🔄 Processing image {idx + 1}/{len(image_filenames)}: {filename}")
        
        image_objects = get_objects_by_filename(filename, ground_truth_json)

        if not image_objects or len(image_objects) < 2:
            print(f"  -> Skipping, found {len(image_objects) if image_objects else 0} objects. Need at least 2.")
            continue

        # Iterate through each object in the image as target_object
        for target_object in image_objects:
            other_objects_in_image = [obj for obj in image_objects if obj != target_object]
            
            # Weighted sampling to get contextual_object
            target_co_occurs = processed_co_data.get(target_object, {})
            context_candidates = [obj for obj in other_objects_in_image if obj in target_co_occurs]
            
            if not context_candidates:
                contextual_object = random.choice(other_objects_in_image)
            else:
                weights = [target_co_occurs[obj] for obj in context_candidates]
                contextual_object = random.choices(context_candidates, weights=weights, k=1)[0]
            
            # Find replacement objects with low co-occurrence frequency with contextual_object
            context_co_occurs = processed_co_data.get(contextual_object, {})
            co_occurring_set = set(context_co_occurs.keys())
            
            # Prefer objects with no co-occurrence records
            zero_co_occurrence_objs = list(all_possible_objects - co_occurring_set - {contextual_object})
            
            replacement_candidates = []
            if len(zero_co_occurrence_objs) >= 8:
                replacement_candidates = random.sample(zero_co_occurrence_objs, 8)
            else:
                replacement_candidates.extend(zero_co_occurrence_objs)
                # Sort by co-occurrence frequency from low to high
                sorted_low_co_occurs = sorted(context_co_occurs.items(), key=lambda item: item[1])
                
                # Fill up to 8 candidates
                for obj, _ in sorted_low_co_occurs:
                    if len(replacement_candidates) >= 8:
                        break
                    if obj not in replacement_candidates:
                        replacement_candidates.append(obj)
            
            formatted_obj_list = str(replacement_candidates).replace('[', '[').replace(']', ']')
            formatted_question = Q_TEMPLATE.format(target_object=target_object, list=formatted_obj_list)

            ori_image_path = IMG_PATH.format(image_name=filename)
            mask_path = MASK_PATH.format(image_name=filename, target_object=target_object)
            image = Image.open(ori_image_path)
            drawed_image = draw_mask(ori_image_path,mask_path)
            drawed_image_base64 = pil_image_to_base64(drawed_image, format='PNG')

            response = send_message(formatted_question, drawed_image_base64)
            response = response.lower()


            for obj in replacement_candidates:
                if obj in response:
                    counterfactual_object = obj
                    break
            
            # Create and save information block
            if len(replacement_candidates) == 8:
                info_block = {
                    "image": filename if filename.lower().endswith('.jpg') else filename + '.jpg',
                    "target_object": target_object,
                    "contextual_object": contextual_object,
                    "counterfactual_object": counterfactual_object
                }
                results.append(info_block)


            try:
                with open(save_json, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)
                print(" Process finished successfully!")
            except Exception as e:
                print(f" Error: Failed to save JSON file. {e}")
    


