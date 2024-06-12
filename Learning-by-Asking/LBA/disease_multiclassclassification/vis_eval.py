import matplotlib.pyplot as plt
import numpy as np

# Precision과 Recall 값
precision = [0.5939, 0.4941, 0.2512, 0.5010, 0.5815, 0.6454]
recall = [0.5708, 0.5085, 0.2923, 0.5560, 0.6145, 0.6552]
classes = ['ResNet50 with finetuning', 
           'ResNet50 with linear probing', 
           'DINOv2 with finetuning', 
           'DINOv2 with linear probing', 
           'DINOv2 with a Rein adapter', 
           'DINOv2 ensembled with two Rein adapter']

colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))  # 색상 배열 생성

# 그래프 그리기
plt.figure(figsize=(10, 8))

for i, (prec, rec, color) in enumerate(zip(precision, recall, colors)):
    plt.scatter(prec, rec, color=color, s=100)  # 포인트 추가
    plt.plot([0, prec], [0, rec], label=classes[i], color=color)  # 원점에서 포인트까지 선 그리기

plt.title('Precision-Recall Curve')
plt.xlabel('Precision(Specificity)')
plt.ylabel('Recall(Sensitivity)')
plt.xlim(0.0, 0.7)
plt.ylim(0.0, 0.7)
plt.legend(loc='best')
plt.grid(True)

# 그래프를 이미지 파일로 저장
plt.savefig("precision_recall_curve.png", format='png', dpi=300)




