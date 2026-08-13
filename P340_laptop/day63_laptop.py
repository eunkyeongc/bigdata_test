# ===========================================================
# 202. 08. 06.  
# 머신러닝실습(회귀_연습문제)
# Page 340.  chapter 08, Section 02, 노트북 가격 예측

#  - 제공된 데이터 목록:
#  - 예측할 컬럼: price

# 제출 파일은 다음 1개의 컬럼을 포함해야 한다.
#  - pred: 예측값(가격)
#  - 제출 파일명: ' result. csv'

# 제출한 모델의 성능은 R2(결정 계수) 평가지표에 따라 채점한다.
# ==========================================================

# 1. 라이브러리 불러오기
import pandas as pd

from sklearn.model_selection import train_test_split # 검증 데이터 나누기
from sklearn.ensemble import RandomForestRegressor # 머신러닝 학습 및 평가
from sklearn.metrics import r2_score


# 2. 파일 불러오기
train = pd.read_csv('P340_laptop/laptop_train.csv')
test = pd.read_csv('P340_laptop/laptop_test.csv')
