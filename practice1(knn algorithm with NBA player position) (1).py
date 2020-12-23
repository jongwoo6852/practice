#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
#수집된 데이터를 분석하기 위해 pandas를 임포트한다.


# In[2]:


df = pd.read_csv("https://raw.githubusercontent.com/wikibook/machine-learning/2.0/data/csv/basketball_stat.csv")
#pandas를 활용해 csb 포맷으로 수집된 데이터를 데이터프레임으로 블러온 후, df라는 변수에 저장한다.


# In[3]:


df.head()
#명령어를 활용해 농구선수 데이터 살펴본다.


# In[4]:


df.Pos.value_counts()
#각 포지션별로 몇 개의 기존 데이터가 있는지 알아본다.


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#데이터 시각화에 필요한 라이브러리 임포트한다.


# In[13]:


sns.lmplot('STL','2P',data=df, fit_reg=False, 
           scatter_kws={"s":150}, 
           markers=["o","x"], 
           hue="Pos")
plt.title('STL and 2P in 2d plane')


# In[14]:


#블로킹, 3점슛, 데이터 시각화
sns.lmplot('BLK','3P', data=df, fit_reg=False, # x축, y축, 데이터, 노 라인
          scatter_kws={"s":150}, # 좌표 상의 점의 크기
          markers=["o", "x"], 
          hue="Pos") #예측값
plt.title('BLK and 3P in 2d plane') #타이틀


# In[15]:


df.drop(['2P','AST', 'STL'], axis=1, inplace = True) #inplace=True는 기존의 데이터 프레임을 덮어쓴다.


# In[16]:


df.head()


# In[17]:


#사이킷런의 train_test_split을 사용하면 코드 한 줄로 손쉽게 데이터를 나눌 수 있습니다.
from sklearn.model_selection import train_test_split


# In[18]:


#다듬어진 데이터에서 20%를 테스트 데이터로 분류합니다.
train, test = train_test_split(df, test_size=0.2)


# In[20]:


train.shape[0]
test.shape[0]


# In[23]:


#kNN 라이브러리 추가
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

#최적의 k를 찾기 위해 교차 검증을 수행할 k의 범위를 3부터 학습 데ㅣ터 절반까지 지정
max_k_range = train.shape[0]//2
k_list = []
for i in range(3, max_k_range, 2):
    k_list.append(i)
    
cross_validation_scores = []
x_train = train[['3P', 'BLK', 'TRB']]
y_train = train[['Pos']]

#교차 검증(10-fold)을 각 k를 대상으로 수행해 검증 결과를 저장
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train.values.ravel(), cv=10, scoring = 'accuracy')
    cross_validation_scores.append(scores.mean())
    
cross_validation_scores


# In[24]:


#k에 따른 정확도를 시각화(한 눈에 볼 수 있도록 데이터 시각화)
plt.plot(k_list, cross_validation_scores)
plt.xlabel('the number of k')
plt.ylabel('Accuracy')
plt.show()


# In[25]:


# 가장 예측율이 높은 k를 선정
k = k_list[cross_validation_scores.index(max(cross_validation_scores))]
print("The best number of k : " +str(k)) #문자열, 최적의 k를 k라는 변수에 저장


# In[26]:


# 라이브러리 임포트
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

Knn = KNeighborsClassifier(n_neighbors=k)

# 학습에 사용될 속성을 지정
x_train = train[['3P', 'BLK', 'TRB']]
# 선수 포지션을 예측할 값으로 지정
y_train = train[['Pos']]

#kNN 모델 학습
knn.fit(x_train, y_train.values.ravel())

# 테스트 데이터에서 분류를 위해 사용될 속성을 지정
x_test = test[['3P', 'BLK', 'TRB']]

# 선수 포지션에 대한 정답을 지정
y_test = test[['Pos']]

#테스트 시작
pred = knn.predict(x_test)

#모델 예측 정확도 출력
print("accuracy :" +str( accuracy_score(y_test.values.ravel(),pred)))

#최저 0.82에서 최대 1.0까지의 모델 정확도를 확인했다.


# In[27]:


comparison = pd.DataFrame({'prediction':pred, 'ground_truth':y_test.values.ravel()})
comparison


# In[ ]:




