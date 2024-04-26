import os
import json

json_file_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/annotations/instances.json'
image_folder_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/images'

with open(json_file_path, 'r') as f:
    data = json.load(f)

id_to_filename = {img['id']: img['file_name'] for img in data['images']}

for img_id, filename in id_to_filename.items():
    new_filename = f"{img_id}_{filename}"
    original_file_path = os.path.join(image_folder_path, filename)
    new_file_path = os.path.join(image_folder_path, new_filename)
    os.rename(original_file_path, new_file_path)

print("이미지 파일 이름이 변경되었습니다.")
