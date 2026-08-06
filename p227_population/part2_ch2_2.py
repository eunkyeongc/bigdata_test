"""
1994년 미국의 인구조사
- 나이, 직업, 학력 등으로 연소득이 5만 달러를 초과 (income) 컬럼 예측
- income --> <= 50K or  < 50, 이진분류(참 or 거짓)
- 평가 지표 --> ROC-AUC
"""

# 라이브러리 불러오기
import pandas as pd

# 내용이 많을 경우 '...' 으로 생략된 부분을 보고 싶을때 설정
pd.set_option('display.max_rows', None)     #모든 행
pd.set_option('display.max_columns', None)  #모든 컬럼
pd.set_option('display.width', None)        #줆바꿈 없이 넓게

# 데이터 블러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# print(train.head())
# print(train['income'].value_counts())

# EDA(탐색적데이터분석)는 
# 1. 크기 --> 2. 타입 --> 3. 통계요약(수치/범주 분리) --> 4. 결측치 --> 5. 이상치 --> 6. 타멧(정답) 분포

# # 결측치 확인
# # print(train.isnull().sum()) # age, workclass, occupation, hours.per.week, native.country
# # print(test.isnull().sum()) # workclass occupation navtive.country

# # 자료형 확인
# # print(train.info())
# # print(test.info())

# # 결측치 삭제 --> 결측치가 있는 데이터(행) 전체 삭제 --> dropna() 기본값 axis=0 (행 삭제)
df = train.dropna()
# # print(df.head())
# # print(train.info())
# # print(df.shape)


# # 결측치가 있는 특정 컬럼 기준으로만 행 삭제 --> subset=[...]
df = train.dropna(subset=['native.country', 'workclass'])
# print(df.isnull().sum())

# # 결측치가 있는 컬럼(열) 자체를 삭제 --> axis=1 --> 열 기준
df = train.dropna(axis=1)
# print(df.shape)
# print(df.info())
# print(df.isnull().sum())

# # 결측치가 많은 특정 컬럼을 직접 지정해서 삭제 --> drop
# print(train.shape)
df = train.drop(['native.country', 'workclass'], axis=1)

# ===========================================================
# 결측치 채우기(범주형)
# 최빈값(mode)으로 채우기
m = train['workclass'].mode()[0]    # [0] -> Private, 최빈값으로 채워넣기
# print(m)
train['workclass'] = train['workclass'].fillna(m)

m = train['native.country'].mode()[0]
train['native.country'] = train['native.country'].fillna(m)

# print(train.isnull().sum())

# 결측치를 새로운  카테고리('X')로 만들어서 채우기(최빈값 대체와 달리, "결측이었다는 사실 자체"도 모델이 학습할 수 있게 한다.)
train['occupation'] = train['occupation'].fillna('X')
# print(train.isnull().sum())
# print(train.info())

# ==============================================================================
# 결측치 채우기(수치형) --> fillna(값)
# age 컬럼을 평균값 구하기
value = int(train['age'].mean())    # 사람 나이는 보통 정수로 측정
# print(value)  # 결과 평균값 38

# age 컬럼을 평균값으로 채우기
train['age'] = train['age'].fillna(value)

# 주당근무시간은 이상치에 덜 민감한 중앙값(median)구하기
value = int(train['hours.per.week'].median())
# print(value)  # 결과 중앙값 40

# 주당근무시간은 이상치에 덜 민감한 중앙값(median)으로 채우기
train['hours.per.week'] = train['hours.per.week'].fillna(value)

# print(train.isnull().sum().sum())
# print(test.isnull().sum())

# test 데이터의 결측치는 train에서 구한 값으로 채워야 한다.
# 범주형 --> 최빈값으로 결측치를 채웠다.
# 수치형 --> 나이는 평균값으로 결측치를 채웠다.

test ['workclass'] = test['workclass'].fillna(test['workclass'].mode()[0])
test ['occupation'] = test['occupation'].fillna(test['occupation'].mode()[0])
test ['native.country'] = test['native.country'].fillna(test['native.country'].mode()[0])
test ['age'] = test['age'].fillna(test['age'].mean())
test ['hours.per.week'] = test['hours.per.week'].fillna(test['hours.per.week'].median())

# print(test.isnull().sum())

# ------------------------------------------------------------------------------------------
# 이상치 처리 -->  age < 0 경우가 있는지 확인(이상치 처리 안 함. 삭제 하지 않음.)
# print(train[train['age'] <= 0])
# print('*'*50)
# print(test[test['age'] <= 0])

# age >= 1 인 데이터만 남김다.(train만 삭제 처리)
train = train[train['age'] > 0]
# print(train.head())
# print(train.head())
# print(train.shape)

# ------------------------------------------------------------------------------------------

# 타겟(target, 정답)을 먼저 분리(전처리 과정에서 X와 y를 함께 다루면 혼돈되기 쉽다)
# pop() : 해당 컬럼을 train에서 꺼내면서 동시에 원본 train에서는 제거한다.
y_train =train.pop('income')

# ------------------------------------------------------------------------------------------
# 인코딩(범주형 -> 숫자)
# 원-핫 인코딩
# 판다스의 get_dummies() : 범주형 컬럼들을 0/1로 이루어진 여러 컬럼으로 자동 변환
# train_oh = pd.get_dummies(train)
# test_oh = pd.get_dummies(test)

# print(train.shape, test.shape, train_oh.shape, test_oh.shape) # 결과 : (29301, 15) (3257, 15) (29301, 107) (3257, 102)

# print(train_oh.info())


# ===== 주의!!! =======
# train과 test를 따로 원핫인코딩을 하면 카테고리 값의 종류가 서로 달라 생성되는 더미 컬럼의 개수/이름이 어긋날 수 있다. (예:train에만 특정 국가명이 존재)
# 이를 해결하기 위해 train + test를 합친 뒤에 한번에 원-핫 인코딩을 한 후 다시 분리하는 것이 안전한 방법이다.

data = pd.concat([train, test], axis=0) # 위 아래로 합치기: 위에는  train 자료, 아래에는  test 자료 배치(행 단위로 합치기 axis=0)
# print(data.shape)  # 결과: (32558, 15)

data_oh = pd.get_dummies(data) # 원-핫 인코딩 수행
print(data_oh.shape) # 결과: (32558, 107)
# print(train.shape) # 결과: (29301, 15)

# 다시 분리하기
# iloc[행번호, 열번호]
print(len(train))  # 결과 : 29301
train_oh = data_oh.iloc[:len(train)].copy()  # 0~29300(chd 2930). 0번부터 train 원본의 갯수 만큼만 복사하여 붙여넣기
test_oh = data_oh.iloc[len(train):].copy()  # train의 갯수부터 끝까지 복사하여 붙여넣기

# ----------------------------------------------------------------------------------------------------------------
# 인코딩(범주형 -> 숫자)
# 레이블 인코딩 --> 사이킷런
# print(train.info())
cols = train.select_dtypes(include = 'object').columns
print(cols) # Index(['workclass',  'education',  'marital.status',  'occupation', 'relationship',   'race', 'sex', 'native.country'], dtype ='object')

from sklearn.preprocessing import LabelEncoder
# train --> fit_transform()
# test --> transform()
for col in cols:
    le =LabelEncoder()
    train[col] = le.fit_transform(train[col])  # fit->학습, transform->변환
    test[col] = le.transform(test[col])        # test는 학습을 하면 안 됨. 그래서 변환만 함.

# ----------------------------------------------------------------------------------------------------------------
# 스캐일링
# 수치형 컬럼들을 조정
# print(train.info())
cols =['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

# 여러 스캐일러를 비교 실습을 할 때, 이전 스케일링이 누적되지 않도록 원본을 복사해서 시작 --> 데이터를 매번 새롭게 불러오기 위해 함수로 제작한다.
def get_data():
    train_copy = train.copy()
    test_copy = test.copy()
    return train_copy, test_copy


# min-max scaling : MinMaxScarler(모든 값을 0 ~ 1 사이로 압축)
# train_copy, test_copy = get_data()   #  데이터 불러오는 함수 호출

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# train_copy[cols] = scaler.fit_transform(train_copy[cols])
# test_copy[cols] = scaler.transform(test_copy[cols])
# print(train_copy.head())
# print(test_copy.head())


# 표준화(Z-Score: 평균 0, 표준편차 1인 분포로 변환) : StandardScaler
# train_copy, test_copy = get_data()
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# train_copy[cols] = scaler.fit_transform(train_copy[cols])
# test_copy[cols] = scaler.transform(test_copy[cols])
# print(train_copy.head())
# print(test_copy.head())

# 로버스트 스케일러 RobusterScaler :  중앙값 사분위값(IQR)을 활용, 평균/표준편차 대신 중앙값 기준이라 이상치의 영향을 덜 받는 장점이 있다.
# train_copy, test_copy = get_data()

# from sklearn.preprocessing import RobustScaler
# scaler = RobustScaler()
# train_copy[cols] = scaler.fit_transform(train_copy[cols])
# test_copy[cols] = scaler.transform(test_copy[cols])
# print(train_copy.head())
# print(test_copy.head())

# 랜덤포레슽 / LightGBM 같은 트리 기반 모델은 스케일링이 성능에 큰 영향을 주지 않는 경우가 많아 실전에서는 생략하기도 한다.

# 원-핫 인코딩은 범주 개수만큰 컬럼이 늘어난다. --> 트리 계열이 아니 모델에 적합하다. 순서없는 범주형에 적합하다.
# 레이블 인코딩은 범주를 0, 1, 2, 3,....  숫자로 이름을 짓는다. 컬럼수는 유지가 된다. 랜덤포레스트/LightGBM 트리 기반 모델에서 자주 사용한다.("크기 순서"가 없는데 숫자로 표현되는 한계가 있다.)

# -----------------------------------------------------------------------------------------------------------------------
# 데이터 분할 : 검증데이터 나누기
from sklearn.model_selection import train_test_split

# train데이터를 학습용(X_train)과 검증요(X_val)으로 분할
X_train, X_val, y_train, y_val = train_test_split(train, y_train, test_size=0.2, random_state=0)
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape ) # 결과 : (23440, 15) (5861, 15) (23440,) (5861,)
      
# -----------------------------------------------------------------------------------------------------------------------
# 머신러닝 학습
# 랜덤포레스트 분류
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0)  
rf.fit(X_train, y_train)
# 각 클래스(레이블)에 속한 확률값 반환(2차원 배열)
pred = rf.predict_proba(X_val)
# print(pred[:10])

# 확률 배열의 각 열이 어떤 클래스를 의미하는지 순서 확인
# print(rf.classes_)  # 결과 : ['<=50K' '>50K']

# -----------------------------------------------------------------------------------------------------------------------
# 예측 및 결과 파일 생성해서 CSV 파일로 내보내기
submit = pd.DataFrame({'pred':pred[:, 1]}) # 양성 클래스('>50K')일 확률만 추출
submit.to_csv('result.csv', index=False)
# print(pd.read_csv('result.csv'))  # 결과 확인하기


# -----------------------------------------------------------------------------------------------------------------------
# 분류 모델의 평가지표
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

#최종 예측 레이블
pred = rf.predict(X_val)

# 정확도 : 전체 데이터 중에서 참으로 예측된 데이터 비율 = (TP+FN)/(TP+TN+FP+FN)
accuracy = accuracy_score(y_val, pred)
# print(accuracy) # 결과 : 0.8698174372973895

# F1_Score : 정밀도와 재현율의 조화평균 --> 클래스 불균형 데이터 정확도보다 신뢰할 수 있다.
# 정밀도(양성으로 예측된 것 중에서 실제로 양성인것= PT/(TP+FP)
# 재현율(민감도, 실제 양성인 데이터 중 모델이 양성으로 올바르게 예측한 비율=TP/(TP+FN)
# 특이도(실제로 음성(Negative)인 데이터를 음성으로 정확하게 맞춘 비율= TN/(TN+FP)
f1 = f1_score(y_val, pred, pos_label='>50K')
# print(f1)   # 결과 : 0.6936973103171417

# -----------------------------------------------------------------------------------------------------------------------
# LightGBM : 그라디언트 부스팅 계열, 오답노트, 정형 데이터에서 비교적 랜덤포레스트보다 성능이 뛰어나다.
import lightgbm as lgb

# verbose = -1  : 학습 중 불필요한 로그 메시지는 출력 안 한다.
lgbmc = lgb.LGBMClassifier(random_state=0, verbose=-1)
lgbmc.fit(X_train, y_train)
pred1 = lgbmc.predict_proba(X_val)   # 확률
roc_auc = roc_auc_score(y_val, pred1[:, 1])  # 1 -> 참을 보겠다.
# print(roc_auc)  # 결과 :  0.9277697320193289

pred2 =lgbmc.predict(X_val)
accuracy = accuracy_score(y_val, pred2)
f1 = f1_score(y_val, pred2, pos_label='>50K')
print(accuracy)  #결과 : 0.8771540692714553
print(f1)   # 결과: 0.7154150197628458