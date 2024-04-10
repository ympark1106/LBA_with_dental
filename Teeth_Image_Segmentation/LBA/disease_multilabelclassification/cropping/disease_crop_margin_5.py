import json
from PIL import Image
import os

def find_decay_teeth(json_path, image_root, output_folder, teeth_category_symbols, margin):
    with open(json_path, 'r') as f:
        data = json.load(f)

    teeth_category_ids = [category['id'] for category in data['categories'] if any(symbol in category['name'] for symbol in teeth_category_symbols)]
    teeth_category_ids.append(85)
    print("teeth category_ids:", teeth_category_ids)

    image_ids_category_38 = [ann['image_id'] for ann in data['annotations'] if ann['category_id'] == 38]
    print("Image_ids with category_id 38:", image_ids_category_38)
    print(len(image_ids_category_38))

    for image_id in image_ids_category_38:
        if image_id >= 206:
            bbox_category_38_all = [ann['bbox'] for ann in data['annotations'] if ann['image_id'] == image_id and ann['category_id'] == 38]
            print("Image ID:", image_id, "\nBbox for category_id = 38:", bbox_category_38_all)
            
            for bbox_category_38 in bbox_category_38_all:
                print("Bbox for category_id = 38:", bbox_category_38)

                for tooth_category_id in teeth_category_ids:
                    print("Tooth category_id:", tooth_category_id)
                    bbox_teeth = [ann['bbox'] for ann in data['annotations'] if ann['image_id'] == image_id and ann['category_id'] == tooth_category_id]
                    print("Bbox teeth:", bbox_teeth)

                    for bbox_tooth in bbox_teeth:
                        bbox_tooth = bbox_widen(bbox_tooth, margin)
                        print("Bbox tooth:", bbox_tooth)

                        if is_bbox_contained(bbox_tooth, bbox_category_38):
                            image_info = next(item for item in data['images'] if item['id'] == image_id)

                            image_path = os.path.join(image_root, image_info['file_name'])
                            output_path = os.path.join(output_folder, f"{image_id}_{tooth_category_id}_{bbox_tooth[0]}_{bbox_tooth[1]}.png")

                            print("Cropping and saving:", output_path)

                            crop_and_save(image_path, output_path, bbox_tooth)


def bbox_widen(bbox, margin):
    x, y, width, height = bbox
    x -= margin
    y -= margin
    width += 2 * margin
    height += 2 * margin

    x = max(0, x)
    y = max(0, y)

    return x, y, width, height


def is_bbox_contained(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    condition_left = x1 <= x2
    condition_right = x1 + w1 >= x2 + w2
    condition_below = y1 <= y2
    condition_above = y1 + h1 >= y2 + h2

    result = condition_left and condition_right and condition_below and condition_above

    print("is_bbox_contained result:", result)
    return result

def crop_and_save(image_path, output_path, bbox):
    x, y, width, height = bbox

    image = Image.open(image_path)
    cropped_image = image.crop((x, y, x + width, y + height))
    cropped_image.save(output_path)

if __name__ == "__main__":
    root = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/new_panorama_coco_dataset/images"
    json_path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/new_panorama_coco_dataset/annotations/instances.json"
    output_folder = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_images/cropped_K05_images_margin150"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    teeth_category_symbols = ['#', 'teeth3']  
    margin = 150

    find_decay_teeth(json_path, root, output_folder, teeth_category_symbols, margin)

cropped_image_path = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_images/cropped_K05_images_margin150'
print("Number of cropped images:", len(os.listdir(cropped_image_path)))