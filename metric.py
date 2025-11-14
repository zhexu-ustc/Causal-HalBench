import json
import os
from collections import defaultdict

def read_json(file_path):
    """读取JSON文件并返回解析后的数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_base_name(image_name):
    """提取基础图像名（去除后缀编号）"""
    base, ext = os.path.splitext(image_name)
    parts = base.split('_')
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 3:
        base = '_'.join(parts[:-1])
    return base + ext

def group_by_base_name_and_type(data):
    """按基础图像名和类型分组数据"""
    groups = defaultdict(lambda: defaultdict(list))
    for item in data:
        base_name = extract_base_name(item['image_name'])
        groups[base_name][item['type']].append(item)
    return groups

def preprocess_data(qa_data, resp_data):
    """预处理数据，构建(基础名, id)到数据项的映射"""
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
    """计算target类型下的准确率和CAC"""
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
    """计算absent类型下的准确率和NAC"""
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
    """计算inpaint类型的错误率CHR"""
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
    计算CPR指标：分别返回target和absent类型的CPR
    配对规则：基础图像（如A.jpg）与每个带编号图像（如A_001.jpg）形成独立配对
    """
    # 辅助函数：判断是否为基础图像（无三位数字后缀）
    def is_base_image(name):
        base, ext = os.path.splitext(name)
        parts = base.split('_')
        return not (len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 3)
    
    # 构建基础图像映射：{base_name: base_image_data}
    base_image_map = {}
    for item in qa_data + resp_data:
        if is_base_image(item['image_name']):
            base_image_map[item['image_name']] = item
    
    # 生成所有有效配对（基础图像-带编号图像）
    pairs = []
    for item in qa_data + resp_data:
        if not is_base_image(item['image_name']):
            base_name = extract_base_name(item['image_name'])
            if base_name in base_image_map:
                pairs.append((base_name, item['image_name']))
    
    # 去重并转换为集合便于处理
    unique_pairs = set(pairs)
    total_target_pairs = total_absent_pairs = 0
    correct_target_pairs = correct_absent_pairs = 0
    
    # 遍历每个独立配对
    for base_name, num_name in unique_pairs:
        # 提取配对中的基础图像和带编号图像数据
        qa_base = [item for item in qa_data if item['image_name'] == base_name]
        qa_num = [item for item in qa_data if item['image_name'] == num_name]
        resp_base = [item for item in resp_data if item['image_name'] == base_name]
        resp_num = [item for item in resp_data if item['image_name'] == num_name]
        
        # 跳过数据不完整的配对
        if not qa_base or not qa_num or not resp_base or not resp_num:
            continue
        
        # 合并数据以便检查标签和类型
        qa_items = qa_base + qa_num
        resp_items = resp_base + resp_num
        
        # ------------------- 处理target类型 -------------------
        # 检查是否同时存在origin和inpainted标签
        has_qa_origin_t = any(item['tag'] == 'origin' and item['type'] == 'target' for item in qa_items)
        has_qa_inpainted_t = any(item['tag'] == 'inpainted' and item['type'] == 'target' for item in qa_items)
        has_resp_origin_t = any(item['tag'] == 'origin' and item['type'] == 'target' for item in resp_items)
        has_resp_inpainted_t = any(item['tag'] == 'inpainted' and item['type'] == 'target' for item in resp_items)
        
        if has_qa_origin_t and has_qa_inpainted_t and has_resp_origin_t and has_resp_inpainted_t:
            total_target_pairs += 1
            # 查找对应的origin和inpainted项
            qa_origin_t = next((item for item in qa_items if item['tag'] == 'origin' and item['type'] == 'target'), None)
            qa_inpainted_t = next((item for item in qa_items if item['tag'] == 'inpainted' and item['type'] == 'target'), None)
            resp_origin_t = next((item for item in resp_items if item['tag'] == 'origin' and item['type'] == 'target'), None)
            resp_inpainted_t = next((item for item in resp_items if item['tag'] == 'inpainted' and item['type'] == 'target'), None)
            
            # 检查回答是否都正确
            if all([qa_origin_t, qa_inpainted_t, resp_origin_t, resp_inpainted_t]):
                origin_correct = qa_origin_t['answer'] == resp_origin_t['answer']
                inpainted_correct = qa_inpainted_t['answer'] == resp_inpainted_t['answer']
                if origin_correct and inpainted_correct:
                    correct_target_pairs += 1
        
        # ------------------- 处理absent类型 -------------------
        has_qa_origin_a = any(item['tag'] == 'origin' and item['type'] == 'absent' for item in qa_items)
        has_qa_inpainted_a = any(item['tag'] == 'inpainted' and item['type'] == 'absent' for item in qa_items)
        has_resp_origin_a = any(item['tag'] == 'origin' and item['type'] == 'absent' for item in resp_items)
        has_resp_inpainted_a = any(item['tag'] == 'inpainted' and item['type'] == 'absent' for item in resp_items)
        
        if has_qa_origin_a and has_qa_inpainted_a and has_resp_origin_a and has_resp_inpainted_a:
            total_absent_pairs += 1
            # 查找对应的origin和inpainted项
            qa_origin_a = next((item for item in qa_items if item['tag'] == 'origin' and item['type'] == 'absent'), None)
            qa_inpainted_a = next((item for item in qa_items if item['tag'] == 'inpainted' and item['type'] == 'absent'), None)
            resp_origin_a = next((item for item in resp_items if item['tag'] == 'origin' and item['type'] == 'absent'), None)
            resp_inpainted_a = next((item for item in resp_items if item['tag'] == 'inpainted' and item['type'] == 'absent'), None)
            
            # 检查回答是否都正确
            if all([qa_origin_a, qa_inpainted_a, resp_origin_a, resp_inpainted_a]):
                origin_correct = qa_origin_a['answer'] == resp_origin_a['answer']
                inpainted_correct = qa_inpainted_a['answer'] == resp_inpainted_a['answer']
                if origin_correct and inpainted_correct:
                    correct_absent_pairs += 1
    
    # 计算CPR指标
    cpr_target = correct_target_pairs / total_target_pairs if total_target_pairs > 0 else 0.0
    cpr_absent = correct_absent_pairs / total_absent_pairs if total_absent_pairs > 0 else 0.0
    
    return {
        'cpr_target': cpr_target,
        'cpr_absent': cpr_absent
    }

def evaluate(qa_file, resp_file):
    """整合所有指标计算的主函数"""
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
        print(f"评估出错: {str(e)}")
        return {}

def metric(qa_file, resp_file):
    results = evaluate(qa_file, resp_file)
    print("评估结果:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

