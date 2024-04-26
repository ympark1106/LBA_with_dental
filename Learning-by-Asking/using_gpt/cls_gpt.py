import os 
import openai
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import requests
import glob
import matplotlib.pyplot as plt
import base64

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
# parent_dir = "/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_images/margin150"
# # category = ['cropped_K00_images', 'cropped_K01_images', 'cropped_K02_images', 'cropped_K03_images', 'cropped_K04_images', 
# #               'cropped_K05_images', 'cropped_K07_images', 'cropped_K08_images', 'cropped_K09_images']
# category = 'cropped_K01_images'
# image_path = glob.glob(os.path.join(parent_dir, category, '*.png'))

# image_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_images/margin150/cropped_K01_images/231_22_1854.0_488.0.png'
# image_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_images/margin150/cropped_K00_images/1436_0_1194.0_541.0.png'
# image_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_images/margin150/cropped_K07_images/1503_0_1224.0_362.0.png'

image_path_global = '/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/images/1368_신경치료-159.jpg'
imaage_path_local = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_images/margin150/cropped_K00_images/1368_8_1270.0_403.0.png'
base64_image_global = encode_image(image_path_global)
base64_image_local = encode_image(imaage_path_local)

# query = "이 이미지는 파노라믹 이미지 내의 일부를 랜덤하게 크롭한 이미지야. 맹출장애라고 의심되는 부분을 모두 찾아서 알려줘. 맹출장애를 진단할 수 있는 몇가지 시각적 특징을 알려줄게. 정상적인 치아 배열에서 벗어난 위치에 있는 치아를 주목해야해. 특히, 치아가 지나치게 기울어져 있거나 다른 치아에 의해 눌려 있는 경우 맹출장애를 의심할 수 있어. 그리고 치아가 나와야 할 정상적인 잇몸 라인 아래에 위치한 치아도 맹출장애를 의심할 수 있어. 치아가 잇몸 또는 뼈 속에 묻혀 있을 때 이러한 현상이 나타나. 그리고 치아가 입안으로 나오는 대신 옆으로 기울어져 있거나 완전히 반대 방향으로 나있는 경우도 맹출장애를 의심할 수 있어. 이는 치아가 맹출을 시도하다가 잘못된 방향으로 성장했음을 나타내. 알려준 특징을 참고해서 의심되는 부분이 없으면 없다고 알려주면 되고 존재하면 몇개가 의심되는지와 의심되는 위치와 이유를 설명해줘. 의심가는 부분의 위치를 설명할 때는 크롭된 이미지에서 내가 보는 방향을 기준으로 왼쪽, 오른쪽, 위, 아래를 명확히 구분해서 알려줘."
query = "전체 파노라믹 이미지와 해당 파노라믹 이미지 내의 일부를 랜덤하게 크롭한 이미지를 보고 크롭된 이미지 내에서 맹출장애라고 의심되는 부분을 모두 찾아서 알려줘. 먼저 크롭된 이미지에서 크롭된 경계 부분을 인지해 잘못 진단하지 않도록 전체 파노라믹 이미지와 크롭된 이미지를 비교해서 크롭된 이미지가 전체 파노라믹 이미지의 어느 위치에 있는지 파악해. 맹출장애를 진단할 수 있는 몇가지 시각적 특징을 알려줄게. 어느정도 기울어져 있는 치아는 정상으로 볼 수 있지만, 너무 많이 기울어져 다른 치아를 간섭하거나 다른 치아에 의해 눌려 있는 경우 맹출장애를 의심할 수 있어. 그리고 치아가 나와야 할 정상적인 잇몸 라인 아래에 위치한 치아도 맹출장애를 의심할 수 있어. 치아가 잇몸 또는 뼈 속에 묻혀 있을 때 이러한 현상이 나타나. 그리고 치아가 입안으로 나오는 대신 옆으로 기울어져 있거나 완전히 반대 방향으로 나있는 경우도 맹출장애를 의심할 수 있어. 이는 치아가 맹출을 시도하다가 잘못된 방향으로 성장했음을 나타내. 알려준 특징을 참고해서 의심되는 부분이 없으면 없다고 알려주면 되고 존재하면 몇개가 의심되는지와 의심되는 위치와 이유를 설명해줘. 의심가는 부분의 위치를 설명할 때는 크롭된 이미지에서 내가 보는 방향을 기준으로 왼쪽, 오른쪽, 위, 아래를 명확히 구분해서 알려줘."

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
 
payload = {
    "model": "gpt-4-turbo",
    "messages": [
      {
        "role": "system", "content": "assistant는 파노라믹 이미지를 통해 치과 질환을 진단하는 치과 의사이다."
        },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": query
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image_global}"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image_local}"
            }
          }
        ]
      }
    ],
    "max_tokens": 500
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
content = response.json()['choices'][0]['message']['content']
# content = response.json()

print(content)

