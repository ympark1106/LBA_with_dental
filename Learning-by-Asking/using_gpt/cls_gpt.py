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
image_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_images/margin150/cropped_K07_images/1503_0_1224.0_362.0.png'
image_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/images/K00-12.jpg'

base64_image = encode_image(image_path)

#query = "파노라믹 이미지의 크롭된 부분을 보고 맹출장애 질환을 가지고 있는지 판별해줘. 먼저 크롭된 부분이 전체 파노라믹 이미지에서 어떤 위치인지 파악하고 크롭된 부분에서 가장자리 부분은 잘려있으니 확실하지 않으면 질병이 없다고 판단하도록 해. 파노라믹 이미지에서 맹출장애 질환의 존재 여부를 판단하는 여러 팁을 줄게. 먼저 치아는 주변과 구별되는 흰색의 덩어리를 치아라고 판단하면 되고, 한 치아가 통상 옆의 다른 치아와 비슷한 형태와 크기를 가지고 있는데 그렇지 않고 잇몸 속에 파묻혀 있거나 정상 치아의 뿌리 부분에 있다면 그 부분을 맹출장애라고 의심하면 돼. 알려준 팁을 참고해서 맹출장애 질환을 가지고 있는지 판별해줘. 그리고 판단한 이유를 설명해줘"
#query = "파노라믹 이미지를 보고 맹출질환을 가지고 있는지 판별해줘. 파노라믹 이미지에서 맹출장애 질환의 존재 여부를 판단하는 여러 팁을 줄게. 먼저 치아는 주변과 구별되는 흰색의 덩어리를 치아라고 판단하면 되고, 한 치아가 통상 옆의 다른 치아와 비슷한 형태와 크기를 가지고 있는데 그렇지 않고 잇몸 속에 파묻혀 있거나 정상 치아의 뿌리 부분에 있다면 그 부분을 맹출장애라고 의심하면 돼. 이미지 하나에 맹출장애 질환이 여러개 존재할 수도 있고 아예 없을 수도 있으니 사진을 차근차근히 보고 있는지 판단해봐. 만약 존재한다면 어떤 위치에 존재하는지와 판단한 이유를 같이 설명해줘"
query = "파노라믹 이미지 내에 맹출장애라고 의심되는 부분을 모두 찾아서 알려줘. 주로 사랑니 위치에 존재하고 그 외에도 치아는 주변과 구별되는 흰색의 덩어리를 치아라고 판단하면 되는데 한 치아가 통상 옆의 다른 치아와 비슷한 형태와 크기를 가지고 있는데 그렇지 않고 잇몸 속에 파묻혀 있거나 정상 치아의 뿌리 부분에 있다면 그 부분을 맹출장애라고 의심하면 돼. 없으면 없다고 알려주면 되고 존재하면 몇개가 의심되는지와 의심되는 위치와 이유를 설명해줘"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
 
payload = {
    "model": "gpt-4-turbo",
    "messages": [
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
              "url": f"data:image/jpeg;base64,{base64_image}"
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