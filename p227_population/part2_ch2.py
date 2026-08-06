# 라이브러리 불러오기
import pandas as pd

# 데이터 블러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.head())
print('='*200)
print(test.head())
print('='*200)
print(train.info())
print('='*200)
print(test.info())
print("="*200)
print(train.shape, test.shape)
print("="*200)
print(train.describe())
print(test.describe())
print("="*200)
print(train.describe(include='all'))
print("="*200)
print(test.describe(include='all'))
print("="*200)
print(train.describe(include='O'))
print("="*200)
print(test.describe(include='O'))
print("="*10, '결측치', "="*10)
print(train.isnull().sum())
print("="*200)
print(test.isnull().sum())

# 결측치 처리, 이상치 확인, 인코딩, 스케일링
# 머신러닝 모델 선정

