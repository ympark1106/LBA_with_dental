import os
import shutil

def remove_duplicate_images(root_dir):
    seen_files = {}
    removal_count = {}
    
    # 모든 하위 디렉토리를 순회합니다.
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # 파일 경로를 구성합니다.
            file_path = os.path.join(subdir, file)
            
            # 이미 처리된 파일인지 확인합니다.
            if file in seen_files:
                # 중복 파일 삭제
                os.remove(file_path)
                # 삭제 카운트 증가
                if file in removal_count:
                    removal_count[file] += 1
                else:
                    removal_count[file] = 1
            else:
                # 최초 등장한 파일이므로 추적합니다.
                seen_files[file] = file_path
    
    # 삭제된 파일과 횟수를 출력합니다.
    for file, count in removal_count.items():
        print(f"{file}: {count}개 삭제됨.")

# 실행 예제
root_dir = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_images/margin150_unique'  # 데이터셋이 위치한 최상위 디렉토리 경로
remove_duplicate_images(root_dir)
