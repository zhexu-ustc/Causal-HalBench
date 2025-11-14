from Intervention_objects_selection import Intervention_objects_selection
from Counterfactual_image_generation import Counterfactual_image_generation


save_json = "image_info.json" # A file to save the generated information blocks
ori_img_txt = "ori_image_name.txt" # A file that records the names of original pictures to be processed
co_count_json = "object_co_count.json" # A file that records co-occurrence information
ground_truth_json = "coco_ground_truth_segmentation.json" # A file that records the ground truth objects in the images
output_dir = "result" # A folder for saving generated results


if __name__ == 'main':
    Intervention_objects_selection(save_json, ori_img_txt, co_count_json, ground_truth_json)
    Counterfactual_image_generation(save_json, ground_truth_json, output_dir)