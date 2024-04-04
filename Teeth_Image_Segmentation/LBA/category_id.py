import json

with open('/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/panorama_dataset/Group1/annotations/instances.json', 'r') as json_file:
    data = json.load(json_file)

for image_id in set(ann['image_id'] for ann in data['annotations'] if ann['category_id'] == 12):
    print(f"Image ID: {image_id}")
    

    bbox_category_12 = [ann['bbox'] for ann in data['annotations'] if ann['image_id'] == image_id and ann['category_id'] == 12]
    
    for bbox in bbox_category_12:
        print(f"Category ID 12 Bbox: {bbox}")

    print("\n")
