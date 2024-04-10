import os
import glob

# folder_path = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_images/duplicated_cropped_K09_images'

# file_parts_count = {}
# to_delete = []

# for file_path in glob.glob(os.path.join(folder_path, '*.png')):
#     file_name = os.path.basename(file_path)
#     parts = file_name.split('_')[-2:]
#     parts_key = '_'.join(parts)
    
#     if parts_key in file_parts_count:
#         file_parts_count[parts_key] += 1
#         to_delete.append(file_path)  
#     else:
#         file_parts_count[parts_key] = 1

# for parts_key, count in file_parts_count.items():
#     if count > 1:
#         print(f'{parts_key}: {count} times')

# print(to_delete)

# for file_path in to_delete:
#     os.remove(file_path)

# len(to_delete)
        

import os
import glob

folder_path = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_images/margin90/cropped_K02_images'

file_parts_count = {}
to_delete = []

sorted_files = sorted(glob.glob(os.path.join(folder_path, '*.png')), key=lambda x: os.path.basename(x))

for file_path in sorted_files:
    file_name = os.path.basename(file_path)
    parts = '_'.join(file_name.split('_')[-2:])
    
    if parts not in file_parts_count:
        file_parts_count[parts] = file_path
    else:
        to_delete.append(file_path)

for file_path in to_delete:
    os.remove(file_path)
    print(f"Deleted: {file_path}")

print(f"Total deleted files: {len(to_delete)}")
print(f"Total remaining files: {len(file_parts_count)}")



