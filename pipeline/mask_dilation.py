import json
import os
from PIL import Image
import numpy as np
import cv2

IMG_PATH = "COCO/val2014/{image_name}.jpg"
MASK_PATH = "COCO/mask/{image_name}/{object}.png"
SAVE_MASK_PATH = "COCO/dilated_mask/{image_name}/{object}.png" 
DILATE_KERNEL_SIZE = 5  # The size of the dilation kernel, must be odd
DILATE_ITERATIONS = 10   # The number of dilation iterations




def mask_dilation(save_json):
    with open(save_json, 'r', encoding='utf-8') as f:
        results = json.load(f)
        
    for idx, item in enumerate(results):
        print(f"🔄 Processing mask {idx + 1}/{len(results)}: {item['image']}-{item['target_object']}")

        if "prompt" in item:
            continue
        
        image_name = item['image']
        target_object = item['target_object']
        contextual_object = item['contextual_object']

        
        # mask_path
        target_obj_mask_path = MASK_PATH.format(image_name=image_name, object=target_object)
        contextual_obj_mask_path = MASK_PATH.format(image_name=image_name, object= contextual_object)
        
        # Load target object mask
        if not os.path.exists(target_obj_mask_path):
            print(f"Skipping {image_name}-{target_object}: Target object mask not found at {target_obj_mask_path}")
            continue
        try:
            target_obj_mask = Image.open(target_obj_mask_path).convert('L') # Convert to grayscale
        except Exception as e:
            print(f"Error opening target mask {target_obj_mask_path}: {e}")
            continue

        # Load contextual object mask (if it exists)
        contextual_mask_array = None
        if contextual_object: # Check if contextual_object is defined
            contextual_obj_mask_path = MASK_PATH.format(image_name=image_name, object=contextual_object)
            if os.path.exists(contextual_obj_mask_path):
                try:
                    contextual_obj_mask = Image.open(contextual_obj_mask_path).convert('L')
                    contextual_mask_array = np.array(contextual_obj_mask)
                    contextual_mask_array[contextual_mask_array > 0] = 255 # Binarize
                    contextual_mask_array[contextual_mask_array <= 0] = 0
                except Exception as e:
                    print(f"Error opening contextual mask {contextual_obj_mask_path}: {e}")
                    # Continue without contextual mask if there's an error
            else:
                print(f"Contextual object mask not found for {image_name}-{contextual_object} at {contextual_obj_mask_path}. Proceeding without it.")
        
        # Convert target mask to numpy array and binarize
        target_mask_array = np.array(target_obj_mask)
        target_mask_array[target_mask_array > 0] = 255
        target_mask_array[target_mask_array <= 0] = 0

        # --- Perform dilation ---
        kernel = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)
        dilated_target_mask = cv2.dilate(target_mask_array, kernel, iterations=DILATE_ITERATIONS)

        # --- Handle overlap with contextual mask ---
        final_inpainting_mask_array = None
        if contextual_mask_array is not None:
            # Find overlap region
            overlap_mask = cv2.bitwise_and(dilated_target_mask, contextual_mask_array)

            # Subtract overlap from dilated mask
            # This ensures the dilated part doesn't cover the contextual object
            safe_dilated_mask = np.maximum(0, dilated_target_mask - overlap_mask)
            safe_dilated_mask[safe_dilated_mask > 0] = 255 # Re-binarize after subtraction

            final_inpainting_mask_array = safe_dilated_mask
        else:
            # No contextual mask, so the dilated target mask is the final one
            final_inpainting_mask_array = dilated_target_mask

        final_inpainting_mask_pil = Image.fromarray(final_inpainting_mask_array)

        # --- Save the final dilated mask ---
        save_dir = os.path.dirname(SAVE_MASK_PATH.format(image_name=image_name, object=target_object))
        os.makedirs(save_dir, exist_ok=True)
        
        final_save_path = SAVE_MASK_PATH.format(image_name=image_name, object=target_object)
        final_inpainting_mask_pil.save(final_save_path)

if __name__ == "__main__":
    save_json = ""
    mask_dilation(save_json)
    print("Mask dilation completed.")