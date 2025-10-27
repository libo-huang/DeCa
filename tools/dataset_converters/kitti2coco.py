import os
import json
import glob
from PIL import Image
from tqdm import tqdm
import random  # 导入 random 库


def convert_kitti_to_coco_split(kitti_root,
                                output_train_json,
                                output_val_json,
                                split_ratio=0.5,
                                dataset_split='training'):
    """
    将KITTI格式的数据集转换为COCO格式，并按比例分割为训练集和验证集。

    修改版：
    - 'Car', 'Van', 'Truck', 'Tram' -> 'Car'
    - 'Pedestrian', 'Person_sitting' -> 'Pedestrian'
    - 'Cyclist' -> 'Cyclist'
    - 其他类别 (Misc, DontCare) 将被忽略。

    :param kitti_root: KITTI数据集的根目录 (例如 '.../kitti/')
    :param output_train_json: 训练集COCO JSON的输出路径
    :param output_val_json: 验证集COCO JSON的输出路径
    :param split_ratio: 训练集所占的比例 (例如 0.8 表示 80%)
    :param dataset_split: 'training' 或 'testing'
    """

    # --- 修改开始 ---

    # 1. 定义最终的COCO类别
    # (Car, Pedestrian, Cyclist)
    final_coco_categories_info = [
        {"name": "Car", "supercategory": "vehicle"},
        {"name": "Pedestrian", "supercategory": "person"},
        {"name": "Cyclist", "supercategory": "cyclist"}
    ]

    coco_categories = []
    final_name_to_id = {}
    # 自动分配ID (1, 2, 3...)
    for i, cat_info in enumerate(final_coco_categories_info, 1):
        cat_id = i
        coco_categories.append({
            "id": cat_id,
            "name": cat_info["name"],
            "supercategory": cat_info["supercategory"]
        })
        final_name_to_id[cat_info["name"]] = cat_id

    # 2. 创建从 KITTI 原始类名 -> 最终COCO类别ID 的映射
    #    (例如 'Van' -> 1, 'Person_sitting' -> 2)
    category_map = {
        # 统一映射到 'Car' (ID: 1)
        'Car': final_name_to_id['Car'],
        'Van': final_name_to_id['Car'],
        'Truck': final_name_to_id['Car'],
        'Tram': final_name_to_id['Car'],

        # 统一映射到 'Pedestrian' (ID: 2)
        'Pedestrian': final_name_to_id['Pedestrian'],
        'Person_sitting': final_name_to_id['Pedestrian'],

        # 映射 'Cyclist' (ID: 3)
        'Cyclist': final_name_to_id['Cyclist']
    }

    # 'Misc' 和 'DontCare' 等其他类别将自动被忽略，
    # 因为它们不在 category_map 中。

    print(f"最终COCO类别 (Categories): {coco_categories}")
    print(f"KITTI -> COCO 的ID映射 (Map): {category_map}")

    # --- 修改结束 ---

    base_coco_format = {
        "info": {
            "description": f"KITTI {dataset_split} dataset in COCO format (Merged Classes)",
            "year": 2025,
            "version": "1.0"
        },
        "licenses": [],
        "categories": coco_categories,  # 使用新的 coco_categories
        "images": [],
        "annotations": []
    }

    import copy
    coco_train = copy.deepcopy(base_coco_format)
    coco_val = copy.deepcopy(base_coco_format)
    coco_val["info"]["description"] = f"KITTI {dataset_split} (Validation Split) dataset (Merged Classes)"

    image_dir = os.path.join(kitti_root, dataset_split, 'image_2')
    label_dir = os.path.join(kitti_root, dataset_split, 'label_2')

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"错误: 目录未找到 {image_dir} 或 {label_dir}")
        return

    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))

    random.seed(42)
    random.shuffle(image_files)

    split_index = int(len(image_files) * split_ratio)
    train_image_files = image_files[:split_index]
    val_image_files = image_files[split_index:]

    print(f"总图像数: {len(image_files)}")
    print(f"训练集图像数: {len(train_image_files)}")
    print(f"验证集图像数: {len(val_image_files)}")

    def process_file_list(files_to_process, coco_output_dict, current_label_dir, pbar_desc):
        """
        处理给定的文件列表，并填充COCO字典。
        """
        image_id_counter = 0
        annotation_id_counter = 0

        for img_path in tqdm(files_to_process, desc=pbar_desc):
            img_filename = os.path.basename(img_path)

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except IOError:
                print(f"无法打开图像 {img_path}, 跳过。")
                continue

            # A. 处理图像 (images)
            image_id = image_id_counter
            image_id_counter += 1

            image_entry = {
                "id": image_id,
                "file_name": img_filename,
                "width": width,
                "height": height
            }
            coco_output_dict["images"].append(image_entry)

            # B. 处理标注 (annotations)
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            label_path = os.path.join(current_label_dir, label_filename)

            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split(' ')
                if not parts or len(parts) < 15:
                    continue

                kitti_class_name = parts[0]

                # 关键：检查原始类名是否在我们的映射中
                if kitti_class_name not in category_map:
                    continue  # 忽略 'DontCare', 'Misc' 等

                # 获取 *新的* COCO 类别 ID
                category_id = category_map[kitti_class_name]

                # C. 提取和转换Bbox
                try:
                    x1, y1, x2, y2 = [float(parts[i]) for i in [4, 5, 6, 7]]

                    x1 = max(0.0, x1)
                    y1 = max(0.0, y1)
                    x2 = min(float(width), x2)
                    y2 = min(float(height), y2)

                    coco_w = x2 - x1
                    coco_h = y2 - y1

                    if coco_w <= 0 or coco_h <= 0:
                        continue

                    coco_bbox = [x1, y1, coco_w, coco_h]
                    area = coco_w * coco_h

                    attributes = {
                        "truncated": float(parts[1]),
                        "occluded": int(parts[2]),
                        "alpha": float(parts[3]),
                        "dimensions_3d": [float(p) for p in parts[8:11]],
                        "location_3d": [float(p) for p in parts[11:14]],
                        "rotation_y": float(parts[14])
                    }

                    annotation_entry = {
                        "id": annotation_id_counter,
                        "image_id": image_id,  # 关联到此JSON中的image_id
                        "category_id": category_id,  # 使用映射后的ID
                        "bbox": coco_bbox,
                        "area": area,
                        "iscrowd": 0,
                        "attributes": attributes
                    }
                    coco_output_dict["annotations"].append(annotation_entry)
                    annotation_id_counter += 1

                except (ValueError, IndexError) as e:
                    print(f"解析行失败: {line.strip()} | 错误: {e}")

    process_file_list(train_image_files, coco_train, label_dir, "处理训练集")
    process_file_list(val_image_files, coco_val, label_dir, "处理验证集")

    print("\n--- 训练集 ---")
    print(f"图像: {len(coco_train['images'])}, 标注: {len(coco_train['annotations'])}")
    with open(output_train_json, 'w') as f:
        json.dump(coco_train, f, indent=4)
    print(f"训练集 JSON 已保存到: {output_train_json}")

    print("\n--- 验证集 ---")
    print(f"图像: {len(coco_val['images'])}, 标注: {len(coco_val['annotations'])}")
    with open(output_val_json, 'w') as f:
        json.dump(coco_val, f, indent=4)
    print(f"验证集 JSON 已保存到: {output_val_json}")


# --- 如何使用 ---
if __name__ == "__main__":
    kitti_data_root = "/data4/maihn/Project/Data/ICDA/KITTI/clear"

    # 2. 定义训练集和验证集的输出路径
    output_train_json = "/data4/maihn/Project/Data/ICDA/KITTI/clear/kitti_train_split.json"
    output_val_json = "/data4/maihn/Project/Data/ICDA/KITTI/clear/kitti_val_split.json"

    # 3. 运行转换和分割
    convert_kitti_to_coco_split(
        kitti_data_root,
        output_train_json,
        output_val_json,
        split_ratio=0.5,  # 80% 训练
        dataset_split='training'
    )