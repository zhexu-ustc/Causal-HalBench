import json
import os
from collections import defaultdict

def read_json(file_path):
    """Read JSON file and return parsed data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_base_name(image_name):
    """Extract base image name (remove suffix number)"""
    base, ext = os.path.splitext(image_name)
    parts = base.split('_')
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 3:
        base = '_'.join(parts[:-1])
    return base + ext

def group_by_base_name_and_type(data):
    """Group data by base image name and type"""
    groups = defaultdict(lambda: defaultdict(list))
    for item in data:
        base_name = extract_base_name(item['image_name'])
        groups[base_name][item['type']].append(item)
    return groups

def preprocess_data(qa_data, resp_data):
    """Preprocess data and build mapping from (base name, id) to data items"""
    qa_map = {}
    resp_map = {}
    for item in qa_data:
        key = (extract_base_name(item['image_name']), item['id'])
        qa_map[key] = item
    for item in resp_data:
        key = (extract_base_name(item['image_name']), item['id'])
        resp_map[key] = item
    return qa_map, resp_map

def calculate_target_metrics(qa_map, resp_map):
    """Calculate accuracy and CAC for target type"""
    origin_correct, origin_total = 0, 0
    inpainted_correct, inpainted_total = 0, 0
    
    for key in qa_map:
        if key in resp_map:
            qa_item, resp_item = qa_map[key], resp_map[key]
            if qa_item['type'] == 'target':
                if qa_item['tag'] == 'origin':
                    origin_total += 1
                    if qa_item['answer'] == resp_item['answer']:
                        origin_correct += 1
                elif qa_item['tag'] == 'inpainted':
                    inpainted_total += 1
                    if qa_item['answer'] == resp_item['answer']:
                        inpainted_correct += 1
    
    o_acc = origin_correct / origin_total if origin_total else 0
    i_acc = inpainted_correct / inpainted_total if inpainted_total else 0
    return {
        'target_origin_accuracy': o_acc,
        'target_inpainted_accuracy': i_acc,
        'cac': o_acc - i_acc
    }

def calculate_absent_metrics(qa_map, resp_map):
    """Calculate accuracy and NAC for absent type"""
    origin_correct, origin_total = 0, 0
    inpainted_correct, inpainted_total = 0, 0
    
    for key in qa_map:
        if key in resp_map:
            qa_item, resp_item = qa_map[key], resp_map[key]
            if qa_item['type'] == 'absent':
                if qa_item['tag'] == 'origin':
                    origin_total += 1
                    if qa_item['answer'] == resp_item['answer']:
                        origin_correct += 1
                elif qa_item['tag'] == 'inpainted':
                    inpainted_total += 1
                    if qa_item['answer'] == resp_item['answer']:
                        inpainted_correct += 1
    
    o_acc = origin_correct / origin_total if origin_total else 0
    i_acc = inpainted_correct / inpainted_total if inpainted_total else 0
    return {
        'absent_origin_accuracy': o_acc,
        'absent_inpainted_accuracy': i_acc,
        'nac': i_acc - o_acc
    }

def calculate_chr(qa_map, resp_map):
    """Calculate error rate CHR for inpaint type"""
    error_count, total_count = 0, 0
    for key in qa_map:
        if key in resp_map:
            qa_item, resp_item = qa_map[key], resp_map[key]
            if qa_item['type'] == 'inpaint':
                total_count += 1
                if qa_item['answer'] != resp_item['answer']:
                    error_count += 1
    return {'chr': error_count / total_count if total_count else 0}

def calculate_cpr(qa_data, resp_data):
    """
    Calculate CPR metrics: return CPR for target and absent types respectively
    Pairing rule: base image (e.g., A.jpg) forms an independent pair with each numbered image (e.g., A_001.jpg)
    """
    # Helper function: Check if it is a base image (without three-digit suffix)
    def is_base_image(name):
        base, ext = os.path.splitext(name)
        parts = base.split('_')
        return not (len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 3)
    
    # Build base image mapping: {base_name: base_image_data}
    base_image_map = {}
    for item in qa_data + resp_data:
        if is_base_image(item['image_name']):
            base_image_map[item['image_name']] = item
    
    # Generate all valid pairs (base image - numbered image)
    pairs = []
    for item in qa_data + resp_data:
        if not is_base_image(item['image_name']):
            base_name = extract_base_name(item['image_name'])
            if base_name in base_image_map:
                pairs.append((base_name, item['image_name']))
    
    # Remove duplicates and convert to set for easy processing
    unique_pairs = set(pairs)
    total_target_pairs = total_absent_pairs = 0
    correct_target_pairs = correct_absent_pairs = 0
    
    # Iterate through each independent pair
    for base_name, num_name in unique_pairs:
        # Extract base image and numbered image data in the pair
        qa_base = [item for item in qa_data if item['image_name'] == base_name]
        qa_num = [item for item in qa_data if item['image_name'] == num_name]
        resp_base = [item for item in resp_data if item['image_name'] == base_name]
        resp_num = [item for item in resp_data if item['image_name'] == num_name]
        
        # Skip pairs with incomplete data
        if not qa_base or not qa_num or not resp_base or not resp_num:
            continue
        
        # Merge data to check tags and types
        qa_items = qa_base + qa_num
        resp_items = resp_base + resp_num
        
        # ------------------- Process target type -------------------
        # Check if both origin and inpainted tags exist
        has_qa_origin_t = any(item['tag'] == 'origin' and item['type'] == 'target' for item in qa_items)
        has_qa_inpainted_t = any(item['tag'] == 'inpainted' and item['type'] == 'target' for item in qa_items)
        has_resp_origin_t = any(item['tag'] == 'origin' and item['type'] == 'target' for item in resp_items)
        has_resp_inpainted_t = any(item['tag'] == 'inpainted' and item['type'] == 'target' for item in resp_items)
        
        if has_qa_origin_t and has_qa_inpainted_t and has_resp_origin_t and has_resp_inpainted_t:
            total_target_pairs += 1
            # Find corresponding origin and inpainted items
            qa_origin_t = next((item for item in qa_items if item['tag'] == 'origin' and item['type'] == 'target'), None)
            qa_inpainted_t = next((item for item in qa_items if item['tag'] == 'inpainted' and item['type'] == 'target'), None)
            resp_origin_t = next((item for item in resp_items if item['tag'] == 'origin' and item['type'] == 'target'), None)
            resp_inpainted_t = next((item for item in resp_items if item['tag'] == 'inpainted' and item['type'] == 'target'), None)
            
            # Check if all answers are correct
            if all([qa_origin_t, qa_inpainted_t, resp_origin_t, resp_inpainted_t]):
                origin_correct = qa_origin_t['answer'] == resp_origin_t['answer']
                inpainted_correct = qa_inpainted_t['answer'] == resp_inpainted_t['answer']
                if origin_correct and inpainted_correct:
                    correct_target_pairs += 1
        
        # ------------------- Process absent type -------------------
        has_qa_origin_a = any(item['tag'] == 'origin' and item['type'] == 'absent' for item in qa_items)
        has_qa_inpainted_a = any(item['tag'] == 'inpainted' and item['type'] == 'absent' for item in qa_items)
        has_resp_origin_a = any(item['tag'] == 'origin' and item['type'] == 'absent' for item in resp_items)
        has_resp_inpainted_a = any(item['tag'] == 'inpainted' and item['type'] == 'absent' for item in resp_items)
        
        if has_qa_origin_a and has_qa_inpainted_a and has_resp_origin_a and has_resp_inpainted_a:
            total_absent_pairs += 1
            # Find corresponding origin and inpainted items
            qa_origin_a = next((item for item in qa_items if item['tag'] == 'origin' and item['type'] == 'absent'), None)
            qa_inpainted_a = next((item for item in qa_items if item['tag'] == 'inpainted' and item['type'] == 'absent'), None)
            resp_origin_a = next((item for item in resp_items if item['tag'] == 'origin' and item['type'] == 'absent'), None)
            resp_inpainted_a = next((item for item in resp_items if item['tag'] == 'inpainted' and item['type'] == 'absent'), None)
            
            # Check if all answers are correct
            if all([qa_origin_a, qa_inpainted_a, resp_origin_a, resp_inpainted_a]):
                origin_correct = qa_origin_a['answer'] == resp_origin_a['answer']
                inpainted_correct = qa_inpainted_a['answer'] == resp_inpainted_a['answer']
                if origin_correct and inpainted_correct:
                    correct_absent_pairs += 1
    
    # Calculate CPR metrics
    cpr_target = correct_target_pairs / total_target_pairs if total_target_pairs > 0 else 0.0
    cpr_absent = correct_absent_pairs / total_absent_pairs if total_absent_pairs > 0 else 0.0
    
    return {
        'cpr_target': cpr_target,
        'cpr_absent': cpr_absent
    }

def evaluate(qa_file, resp_file):
    """Main function integrating all metric calculations"""
    try:
        qa_data = read_json(qa_file)
        resp_data = read_json(resp_file)
        qa_map, resp_map = preprocess_data(qa_data, resp_data)
        
        target_metrics = calculate_target_metrics(qa_map, resp_map)
        absent_metrics = calculate_absent_metrics(qa_map, resp_map)
        chr_metrics = calculate_chr(qa_map, resp_map)
        cpr_metrics = calculate_cpr(qa_data, resp_data)
        
        return {
            **target_metrics,
            **absent_metrics,
            **chr_metrics,
            **cpr_metrics
        }
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return {}

def metric(qa_file, resp_file):
    results = evaluate(qa_file, resp_file)
    print("Evaluation results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")