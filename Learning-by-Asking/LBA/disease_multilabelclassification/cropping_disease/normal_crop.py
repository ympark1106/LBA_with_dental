import json
from PIL import Image
import os
import random


def random_crop_teeth(json_path, image_root, output_folder, teeth_symbol, num_crops, margin):
    with open(json_path, 'r') as f:
        data = json.load(f)

    teeth_ids = [category['id'] for category in data['categories'] if teeth_symbol in category['name']]
    print("teeth category_ids:", teeth_ids)

    # image_ids_not_disease = [ann['image_id'] for ann in data['annotations'] if ann['category_id'] != 33 and ann['category_id'] != 34 and ann['category_id'] != 35 and 
    #                          ann['category_id'] != 36 and ann['category_id'] != 37 and ann['category_id'] != 38 and ann['category_id'] != 39 and ann['category_id'] != 40 and ann['category_id'] != 41]

    image_ids_disease = [ann['image_id'] for ann in data['annotations'] if ann['category_id'] in [33, 34, 35, 36, 37, 38, 39, 40, 41]]

    # print("Image_ids not with category_id 34:", image_ids_not_disease)

    for image_id in set(image_ids_disease):
        if image_id >= 206:
            decay_bboxes = [ann['bbox'] for ann in data['annotations'] if ann['image_id'] == image_id and ann['category_id'] in [33, 34, 35, 36, 37, 38, 39, 40, 41]]
            print("Image ID:", image_id, "\nBbox for diseases:", decay_bboxes)

            for decay_bbox in decay_bboxes:
                print("Decay bbox:", decay_bbox)

                for tooth_id in teeth_ids:
                    tooth_bboxes = [ann['bbox'] for ann in data['annotations'] if ann['image_id'] == image_id and ann['category_id'] == tooth_id]
                    print("Tooth bboxes:", tooth_bboxes)

                    for tooth_bbox in tooth_bboxes:
                        tooth_bbox = bbox_widen(tooth_bbox, margin)
                        print("Tooth bbox:", tooth_bbox)

                        if all(not is_bbox_contained(tooth_bbox, decay_bbox) for decay_bbox in decay_bboxes):
                            image_info = next(item for item in data['images'] if item['id'] == image_id)
                            image_path = os.path.join(image_root, image_info['file_name'])
                            output_path = os.path.join(output_folder, f"{image_id}_{tooth_id}_{tooth_bbox[0]}_{tooth_bbox[1]}.png")
                            print("Cropping and saving:", output_path)
                            crop_and_center(image_path, output_path, tooth_bbox)
                


def bbox_widen(bbox, margin):
    x, y, width, height = bbox
    x -= margin
    y -= margin
    width += 2 * margin
    height += 2 * margin
    return max(0, x), max(0, y), width, height

def is_bbox_contained(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return not (x1 > x2 + w2 or x1 + w1 < x2 or y1 > y2 + h2 or y1 + h1 < y2)


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
    output_folder = "/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_disease_images/cropped_normal_images"
    teeth_symbol = '#'
    num_crops = 300
    margin = 50
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    random_crop_teeth(json_path, root, output_folder, teeth_symbol, num_crops, margin)

cropped_image_path = "/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_disease_images/cropped_normal_images"
print("Number of cropped images:", len(os.listdir(cropped_image_path)))