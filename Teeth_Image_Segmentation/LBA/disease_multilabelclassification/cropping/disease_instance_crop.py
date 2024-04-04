import json
from PIL import Image
import os

def find_disease(json_path, image_root, output_folder, margin):
    with open(json_path, 'r') as f:
        data = json.load(f)

    disease_category_ids = [33, 34, 35, 36, 37, 38, 39, 40, 41] 

    total_disease = 0

    for disease_category_id in disease_category_ids:
        ann_disease = [ann for ann in data['annotations'] if ann['category_id'] == disease_category_id]
        print(f"Disease_category_id {disease_category_id}: {len(ann_disease)}")
        total_disease += len(ann_disease)


        for ann in ann_disease:
            bbox = ann['bbox']
            image_id = ann['image_id']
            image_info = next(item for item in data['images'] if item['id'] == image_id)
            # print(image_info)
            image_path = os.path.join(image_root, image_info['file_name'])
            output_path = os.path.join(output_folder, f"{disease_category_id}_{image_id}_{bbox[0]}_{bbox[1]}.png")

            print(f"Cropping and saving: {output_path}")
            crop_and_save(image_path, output_path, bbox, margin)

    print(f"Total images: {total_disease}")   


def crop_and_save(image_path, output_path, bbox, margin):
    x, y, width, height = bbox

    x -= margin
    y -= margin
    width += 2 * margin
    height += 2 * margin

    x = max(0, x)
    y = max(0, y)

    image = Image.open(image_path)
    cropped_image = image.crop((x, y, x + width, y + height))
    cropped_image.save(output_path)

if __name__ == "__main__":
    root = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/new_panorama_coco_dataset/images"
    json_path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/new_panorama_coco_dataset/annotations/instances.json"
    output_folder = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_images/cropped_disease_images"
    margin = 30

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    find_disease(json_path, root, output_folder, margin)

# cropped_image_path = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_images/cropped_disease_images'
# print("Number of cropped images:", len(os.listdir(cropped_image_path)))

# 38 10개 부족