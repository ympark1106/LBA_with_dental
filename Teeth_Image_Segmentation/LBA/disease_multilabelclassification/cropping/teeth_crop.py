import json
from PIL import Image
import os
import random

'''
cropped_image_path = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/not_K02_images'
print("Number of cropped images:", len(os.listdir(cropped_image_path)))
exit()
'''

def random_crop_teeth(json_path, image_root, output_folder, teeth_symbol, num_crops, margin):
    with open(json_path, 'r') as f:
        data = json.load(f)

    teeth_ids = [category['id'] for category in data['categories'] if teeth_symbol in category['name']]
    print("teeth category_ids:", teeth_ids)

    image_ids_not_disease = [ann['image_id'] for ann in data['annotations'] if ann['category_id'] != 33 and ann['category_id'] != 34 and ann['category_id'] != 35 and 
                             ann['category_id'] != 36 and ann['category_id'] != 37 and ann['category_id'] != 38 and ann['category_id'] != 39 and ann['category_id'] != 40 and ann['category_id'] != 41]

    # print("Image_ids not with category_id 34:", image_ids_not_disease)


    print("Teeth image_ids:", image_ids_not_disease)

    for image_id in random.sample(image_ids_not_disease, min(num_crops, len(image_ids_not_disease))):
        if image_id >= 206:
            bboxes = [ann['bbox'] for ann in data['annotations'] if ann['image_id'] == image_id]

            for bbox in random.sample(bboxes, min(num_crops, len(bboxes))):
                image_info = next(item for item in data['images'] if item['id'] == image_id)

                image_path = os.path.join(image_root, image_info['file_name'])
                output_path = os.path.join(output_folder, f"{image_info['id']}_random_cropped_teeth.png")

                print("Cropping and saving:", output_path)

                crop_and_save(image_path, output_path, bbox, margin)

def crop_and_save(image_path, output_path, bbox, margin):

    x, y, width, height = bbox
    x -= margin
    y += margin
    width += 2 * margin
    height -= 2 * margin

    x = max(0, x)
    y = max(0, y)

    image = Image.open(image_path)
    cropped_image = image.crop((x, y, x + width, y + height))
    cropped_image.save(output_path)

if __name__ == "__main__":
    root = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/new_panorama_coco_dataset/images"
    json_path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/new_panorama_coco_dataset/annotations/instances.json"
    output_folder = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/others_images"
    teeth_symbol = '#'
    num_crops = 500
    margin = 0

    random_crop_teeth(json_path, root, output_folder, teeth_symbol, num_crops, margin)

cropped_image_path = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/others_images'
print("Number of cropped images:", len(os.listdir(cropped_image_path)))