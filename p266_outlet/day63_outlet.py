# ===============================================
# 202. 08. 06.  머신러닝실습(회귀)
# ===============================================

"""
p266_outlet/day63.py

빅분기 실기 ch4 작업형 2번 회귀(Regression) 실습

10개의 아울렛 매장에서 1,500여 개의 제품에 대한 판매 데이터를 수집했다. 
예측 모델을 만들고 아울렛 특정 매장에서 각 제품의 판매금액을 예측하시오.
· 평가 기준은 RMSE로 평가 
· label(target）은 판매금액（Item_Outlet_Sale) 
· 제출 파일은 예측값만 result. csv 파일로 생성해 제출（컬럼명 : precl, 1개)

----------------------------------
1. 라이브러리 불러오기
2. 데이터 불러오기
3. 탐색적 데이터 분석(EDA)
4. 데이터 전처리 - 인코딩, 스케일링
5. 검증 데이터 분할
6. 머신러닝 학습 및 평가
7. 예측 및 결과 파일 생성
"""

#------------------------------------------
# 1. 라이브러리 불러오기
#------------------------------------------
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # train과 test 분리
from sklearn.linear_model import LinearRegression # 선형회귀
from sklearn.ensemble import RandomForestRegressor # 랜덤포레스트 회귀
import lightgbm as lgb #  라이트지비엠 라이브러리 불러오기. 별명은 lgb
from sklearn.metrics import root_mean_squared_error # RMSE 평가지표

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


#------------------------------------------
# 2. 데이터 불러오기
#------------------------------------------

train =pd.read_csv('p266_outlet/train.csv')
test = pd.read_csv('p266_outlet/test.csv')
# print('===데이터 확인하기===')
# print(train.head())
# print('\n')
# print(test.head())

# print('\n')
# print('===행과 열의 개수===')
# print(train.shape, test.shape)

# print('\n')
# print('===컬럼과 자료형 확인하기===')
# print(train.info())
# print('\n')
# print(test.info())

# print('\n')
# print('=== 기초 통계량 확인하기(수치형만) ===')
# print(train.describe())
# print('\n')
# print(test.describe())

# print('\n')
# print('=== 기초통계량 확인하기(범주형만) ===')
# print(train.describe(include='O'))
# print('\n')
# print(test.describe(include='O'))

# print('\n')
# print('=== 결측치 확인하기 ===')
# print(train.isnull().sum())
# print('\n')
# print(test.isnull().sum())

# print('\n')
# print('=== 타켓(정답) 변수의 분포(히토그램)를 확인하기 ===')
# print(train['Item_Outlet_Sales'].hist())


#------------------------------------------
# 3. 탐색적 데이터 분석(EDA)
#------------------------------------------

# 범주형(object 컬럼 목록 탐지)
# print(train.columns[train.dtypes == object])

#------------------------------------------
# 4. 데이터 전처리 - 인코딩, 스케일링
#------------------------------------------

# 인코딩 대상 컬럼 지정
cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

# target(매출액)을 먼저 분리
target = train.pop('Item_Outlet_Sales')
# print(target.head())
# print(train.shape, test.shape)

#train + test --> 인코딩
df = pd.concat([train, test])
# print('=== train과 test을 합친 행과 열의 개수 ===')
# print(df.shape)

# 레이블 인코딩 
le = LabelEncoder()

# 인코딩해야할 컬럼들이 6개이므로 반복문을 돌린다.
for col in cols:
    df[col] = le.fit_transform(df[col])

# print('== 숫자형컬럼 인코딩 결과 ===')
# print(df.head())

# 다시 train과 test 분리하기
train = df.iloc[:len(train)].copy()
test = df.iloc[len(train):].copy()

# print('=== 분리된 train과 test의 행과 열의 개수 확인하기 ===')
# print(train.shape, test.shape)

# 결측치 처리하기(Item_Weight, Outlet_Size)

# Item_Weight(상품 무게) : 최솟값으로 채움. min() 사용
train['Item_Weight'] = train['Item_Weight'].fillna(train['Item_Weight'].min())
test['Item_Weight'] = test['Item_Weight'].fillna(test['Item_Weight'].min())

# Outlet_Size(매장 크기) : 최빈값으로 채운다. mode() 사용
train['Outlet_Size'] = train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])
test['Outlet_Size'] = test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0])

# print('=== 결측치 처리 결과 확인하기 ===')
# print(train.isnull().sum())
# print('\n')
# print(test.isnull().sum())

# 예측에 도움이 안되는 식별자(ID) 컬럼 제거 --> axis=1
# print(train.columns)
train = train.drop('Item_Identifier', axis=1)
test = test.drop('Item_Identifier', axis=1)

# print(train.shape, test.shape)

#------------------------------------------
# 5. 검증 데이터 분할
#------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=0)
# print('\n')
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

#------------------------------------------
# 6. 머신러닝 학습 및 평가(선형회귀)
#------------------------------------------
# lr = LinearRegression()
# lr.fit(X_train, y_train) # 학습

# # 회귀는 분류와 달리 predict()가 바로 예측값(숫자)을 반환
# y_pred = lr.predict(X_val)  # 예측
# print(y_pred)

# result = root_mean_squared_error(y_val, y_pred)
# print('===  선형 회귀 예측 결과 ===')
# print(result)

#------------------------------------------
# 6. 머신러닝 학습 및 평가(랜덤포레스트회귀)
#------------------------------------------
rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train) # 학습
y_pred= rf.predict(X_val) # 예측
result = root_mean_squared_error(y_val, y_pred)
print('===  랜덤포레스트 회귀 예측 결과 ===')
print(result)

#------------------------------------------
# 6. 머신러닝 학습 및 평가(LightGBM 회귀)
#------------------------------------------
# #LightGBM 회귀 -  부스팅 계열, 보통 랜덤포레스트보다 성능이 좋은 경우가 많다.
# model = lgb.LGBMRegressor(random_state=0, verbose=-1)
# model.fit(X_train, y_train) # 학습
# y_pred = model.predict(X_val) # 예측
# result = root_mean_squared_error(y_val, y_pred) # RMSE 수치를 구한다. 작을 수록 좋다.
# print('=== LightGBM 회귀 예측 결과 ===')
# print(result)


#------------------------------------------
# 7. 예측 및 결과 파일 생성 : 예측 -> 데이터프레임 -> CSV 파일로 보내기
#------------------------------------------
# 최종 선택한 모델을 실제 test 데이터 예측
pred = rf.predict(test)
print('=== 데이터 예측 결과 ===')
print(pred)

# 데이터프레임으로 생성
submit = pd.DataFrame({'pred': pred})
print('=== 결과물 확인하기 ===')
print(submit.head())

# CSV로 내보내기
submit.to_csv('p266_outlet/result.csv', index=False)

# CSV 파일 확인하기
temp = pd.read_csv('p266_outlet/result.csv')
print('\n')
print('=== result.csv 파일 확인하기 ===')
print(temp.head())





