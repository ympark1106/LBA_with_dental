import json
from PIL import Image
import os

def find_decay_areas(json_path, image_root, output_folder):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_ids_with_decay = [ann['image_id'] for ann in data['annotations'] if ann['category_id'] in [38]]
    print("Image_ids with category_id 38:", image_ids_with_decay)
    print(len(image_ids_with_decay))

    for image_id in set(image_ids_with_decay):
        decay_bboxes = [ann['bbox'] for ann in data['annotations'] if ann['image_id'] == image_id and ann['category_id'] in [38]]
        print("Image ID:", image_id, "\nBbox for category_id = 38:", decay_bboxes)
        


        for bbox in decay_bboxes:
            image_info = next(item for item in data['images'] if item['id'] == image_id)
            image_path = os.path.join(image_root, image_info['file_name'])
            output_path = os.path.join(output_folder, f"{image_id}_{bbox[0]}_{bbox[1]}.png")

            crop_and_center(image_path, output_path, bbox)

def crop_and_center(image_path, output_path, bbox):
    x, y, width, height = bbox
    target_size = 224

    image = Image.open(image_path)
    center_x = x + width // 2
    center_y = y + height // 2

    left = max(0, center_x - target_size // 2)
    top = max(0, center_y - target_size // 2)
    right = min(image.width, center_x + target_size // 2)
    bottom = min(image.height, center_y + target_size // 2)

    cropped_image = image.crop((left, top, right, bottom))
    cropped_image.save(output_path)

if __name__ == "__main__":
    root = "/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/images"
    json_path = "/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/annotations/instances.json"
    output_folder = "/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_disease_images/cropped_K05_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    find_decay_areas(json_path, root, output_folder)

cropped_image_path = "/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_disease_images/cropped_K05_images"
print("Number of cropped images:", len(os.listdir(cropped_image_path)))