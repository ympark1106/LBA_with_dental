#%%
import numpy as np
import torch
# %%
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)
# %%
print('Rank of t: ', t.ndim) #1개의 차원
print('Shape of t: ', t.shape) # 7개의 elements
# %%
print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) #Element
print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1]) # Slicing
print('t[:2] t[3:] = ', t[:2], t[3:]) #Slicing
# %%
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
# %%
#1D Array with PyTorch
print(t.dim()) #rank    
print(t.shape) #shape
print(t.size()) #shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])
# %%
#2D Array with PyTorch
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ]) # 4X3
print(t)
# %%
print(t.dim())
print(t.size())
print(t[:, 1])
print(t[:, 1].size())
print(t[:, :-1])
# %%
#Broadcasting
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)
# %%
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3]]) # 3 -> [[3, 3]]
print(m1 + m2)
# %%
# 2 x 1 Vector + 1 X 2 Vector 
m1 = torch.FloatTensor([[1, 2]])   #느낌이 가로 세로 대칭되게 벡터 자동 늘어남.
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)
# %%
#Matrix Multiplication
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print(m1.matmul(m2)) #행렬곱
print(m1 * m2) #elements끼리 곱
# %%
#Mean
t = torch.FloatTensor([[1, 2]])
print(t.mean())
# %%
#Can't use mean() on integers
t = torch.LongTensor([1, 2])
try:
    print(t.mean())
except Exception as exc:
    print(exc)
# %%
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
# %%
print(t.mean())
print(t.mean(dim=0)) #행 별 연산
print(t.mean(dim=1)) #열 별 연산
print(t.mean(dim=-1)) #마지막 차원 별 연산
# %%
#view
#차원 재구성(메모리 공간은 동일하게)
t = np.array([[[0, 1, 2],
               [3, 4, 5]],


         
              [[6, 7 ,8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)
# %%
# 기존 tensor의 크기와 동일하도록
# -1은 나머지 크기에 따라 자동으로 정해짐.
print(ft.view([-1, 3])) 
print(ft.view([-1, 3]).shape)
# %%
print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)
# %%
#Squeeze
#차원이 1인 차원을 제거해줌
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
# %%
print(ft.squeeze())
print(ft.squeeze().shape) 
# %%
#Unsqueeze
ft = torch.Tensor([0, 1, 2]) #차원 크기 (3)
print(ft.shape)
# %%
print(ft.unsqueeze(0)) # 0 -> dim = 0 과 동일 / 차원 크기 (1, 3)
print(ft.unsqueeze(0).shape)
# %%
print(ft.view(1, -1))
print(ft.view(1, -1).shape)
#%%
print(ft.unsqueeze(1)) # 1 -> dim = 1 과 동일 / 차원 크기 (3, 1)
print(ft.unsqueeze(1).shape)


