# project1-End-Type-Star-Regression

## 1. 프로젝트 개요

이번 프로젝트의 주제는 **'회귀 모델을 사용한 항성 죽음 예측'** 이다.

모든것엔 시작과 끝이 있듯이 무한해 보이는 우주 공간에도 '끝'은 있다. 

우리가 항상 볼 수 있는 태양 또한 먼 미래엔 최후를 맞이 할 수 있다는 뜻이다. 

흔히 지구에서 항성 간의 거리를 '광년' 으로 표기하곤 하는데, 만약 A항성의 거리가 100광년 이라면, 빛의 속도로 100년이 지나야지만 A항성에 도달 할 수 있을것이다.
결국 우리가 보는 A항성의 빛은 100년전의 빛을 보는것이다. 이 원리를 적용하자면 우리가 바라보는 밤하늘의 별빛들은 어쩌면 이미 '죽은' 항성의 빛일지도 모른다는 사실이다.

인간은 호기심의 동물이다. 미래의 상황을 알지 못한채 그 불확실함에 연구를 하고 그로 인해 '예측'을 할 수 있게 됩으로써 실제 상황에 대비하고 시간과 차원의 범위가 증가하게 된다.

그렇기에 이 주제를 선택한 것도 위와 같은 이유 때문이다. 지구에서의 과거의 빛을 통해 우주의 현재 혹은 미래를 예측한다는 일은 순수한 탐구심과 미래에 대한 '가능성'을 보고 싶어
다음과 같은 주제를 선택하게 되었다. 

항성은 에너지와 잔해에 의해 태어나 잔해로 돌아간다. 

일반적으론 중력과 복사압이 서로 동일한 크기로 작용하지만, 항성이 진화함에 따라 그 균형이 무너지게 된다. 균형이 무너지면 더 이상 항성으로 진화하지 못하고 죽음의 길을 걷게 된다. 

항성의 질량에 따라 최후의 형태가 달라지며, 이 질량은 태양 질량을 기준으로 계산된다. 

위 본문을 바탕으로 본 프로젝트에선 항성의 온도, 질량, 스펙트럼 유형, 현재 항성의 유형을 고려하여 항성의 최종 형태를 AI를 활용하여 예측하는 것이 목적이다.



---



## 2. 사용 데이터

Train과 Test data를 같은 data를 사용하면 모델 성능을 파악하는데 제대로 알 수 없기 때문에 Test data를 통한 모델 성능을 잘 보기 위해서 일부러 두 data를 다른 data set을 사용하였다.

Train data는 kaggle에서 가져왔다. [kaggle_Train_data](https://www.kaggle.com/datasets/vinesmsuic/star-categorization-giants-and-dwarfs)

본 data는 항성의 유형 중 거성과 왜성을 분류하기 위한 data로 프로젝트에선 이를 활용하여 모델을 학습 시킬때 사용할 것이다.


Test data는 NASA EXOPLANET ARCHIVE에서 가져왔다. [nasa_archive_Test_data](https://exoplanetarchive.ipac.caltech.edu/)

실제 archive에선 항성에 대한 수많은 열들이 있으며, data를 가져올떈 자신이 하고자 하는 목적에 맞추어 열을 골라 가져와야 한다.

필자도 프로젝트의 목적에 부합하기 위해 질량, 온도, 스펙트럼 유형에 관한 열만 가지고 왔다. 

이 두 data에 대한 자세한 전처리 과정은 다음을 참고하면 된다.

[Data pre_processing](https://github.com/kimsyso/project1-End-Type-Star-/raw/main/data_preprocessing.ipynb)


---


## 3. 모델 설명

[Model_Learning](https://github.com/kimsyso/project1-End-Type-Star-/raw/main/model_learning.ipynb)

본 프로젝트의 목적은 '예측'  즉, '회귀' 문제를 다루는 것이기 때문에 회귀 모델을 사용하였으며, 여러 모델 중 Decision Tree, Random Forest, CatBoost, AdaBoost 이 4종류를 사용하여 학습을 진행하였다.

2의 data 전처리 과정 code를 참고하여 각 label들을 encoding시킬때, 특정 조건을 걸어 그에 따라 분류한 뒤 그 결과를 encoding 시켰다. 

조건 하나하나에 대하여 결과값의 분기점이 나뉘어지는 형태를 보면서 제일 먼저 'Decision Tree' 기법이 떠올랐다. 

왜냐하면 Decision Tree도 위와 마찬가지로 특정 조건에 따라 data를 분류하기 때문에 train data의 성질과 알맞다고 판단하여 사용하게 되었다.

이의 개념을 확장시켜 여러 decision tree의 결과를 결합시켜 하나의 결과에 도달하는 Random Forest를 자연스럽게 사용하게 되었다.

2의 전처리 과정을 참고하면 알수 있듯이 일부 열은 수치형이 아닌 범주형 data가 있었고, 이를 모델 학습하기 위해서 각 요소 별로 encoding 했다. 

하지만 아무리 encoding을 진행했어도 범주형 data라는 사실은 변함 없기 때문에 어떤 모델을 사용해야 할지 고민일떄 CatBoost를 발견하게 되었다.

CatBoost는 범주형 변수의 예측 모델에 최적화된 모델이며, 예측 속도가 빠를뿐 아니라 예측력을 높일 수 있는 모델이기에 정답 layer가 범주형 data의 encoding된 값인 train data에 최적의 모델이라 생각이 들어 사용했다.

마지막으로 AdaBoost이다. AdaBoost는 초기에 약한 모형을 만들고 매 단계마다 이 모형의 약점을 보완해서 새로운 모형으로 순차적으로 적합시키는 방법이다.

앞의 세 모델처럼 거창한 이유는 없지만 단지 새로운 모형이 앞의 세 모델보다 강력한 모형이 만들어질지 궁금했기 때문에 사용하였다.

모델 학습을 위한 library는 다음과 같다. 

CatBoost를 사용하기 위해 pip을 통해 설치해주고 최적의 하이퍼 파라미터를 자동으로 찾아주는 optuna기능을 각 모델마다 사용해서 그 값을 구하기 위해서 optuna도 설치해주었다.

그 이외의 나머지 library는 다음과 같다.

- pandas(csv file load)

- sklearn(model, data split, metrics load)

  - train_test_split(data 분할)
 
  - DecisionTreeRegressor
 
  - RandomForestRegressor, AdaBoostRegressor, VotingRegressor(optuna한 결과 voting)
 
  - mean_absolute_error, mean_squared_error, r2_score(모델 성능평가 지표)

- CatBoostRegressor

- optuna(함수, 파라미터 정의)

  - trial(optuna 목적함수의 반복 시도 횟수 설정)
 
  - TPESampler(하이퍼 파라미터 최적화 수행)
 
 - matplotlib.pyplot(model 결과 시각화)

 - numpy



---


## 4. 결과

초반엔 각 모델의 파라미터를 직접 조정하면서 최적의 파라미터를 찾으려 했었다. 

하지만 decision tree를 작업하던 도중 파라미터를 몇을 넣든 최종 평가 지표의 차이가 없다는 것을 발견하고 더 이상 직접 작업하는것엔 의미가 없다 판단하여 최적의 하이퍼 파라미터를 자동으로 찾아주는 프레임 워크인 'optuna' 기능을 사용하였다.

optuna 기능을 사용하여 출력된 각 모델의 최적의 하이퍼 파라미터 값과 그 평가 지표이다. 

이번 프로젝트에 사용한 최종 성능 평가 지표는 MSE, MAE, 그리고 R2 score 이다.


### - Decision Tree

![optuna_decision_tree](https://github.com/kimsyso/project1-End-Type-Star-/blob/main/optuna_decision_tree.png)




### - Random Forest

![optuna_random_forest](https://github.com/kimsyso/project1-End-Type-Star-/blob/main/optuna_random_forest.png)




### - CatBoost regressor

![optuna_catboost](https://github.com/kimsyso/project1-End-Type-Star-/blob/main/optuna_catboost.png)




### - AdaBoost regressor

![optuna_adaboost](https://github.com/kimsyso/project1-End-Type-Star-/blob/main/optuna_adaboost.png)




이후 optuna한 결과를 토대로 최종 평가지표를 구하기 위해 'votingregressor' 기능을 사용하였고,

결과 출력 후 성능이 어떠한지 정확하게 판단하기 위해서 'residual plot' 과 'scatter plot'을 사용하여 시각화 하였다.




### - Voting regressor 

![voting_regressor](https://github.com/kimsyso/project1-End-Type-Star-/blob/main/voting_regressor.png)




위의 voting regressor에 대한 시각화 결과이다.

### - Residual plot(잔차학습)

![residual_plot](https://github.com/kimsyso/project1-End-Type-Star-/blob/main/residual%20plot.png)




### - Scatter plot(산점도)

![scatter_plot](https://github.com/kimsyso/project1-End-Type-Star-/blob/main/scatter.png)




위의 자료를 토대로 총평을 내리자면 optuna 기능을 사용하면서 직접 파라미터를 찾았을때의 결과보다 조금 더 나아지긴 했지만 여전히 성능이 안좋은건 동일하다.

이를 더 확실하게 볼 수 있는것이 시각화 결과 였는데, 잔차학습에서 -3 을 기준으로 잔차를 나누었는데 

예측값이 늘어날수록 잔차가 하향세를 보이고 일부 plot들은 멀리 떨어져 있어 모델 예측이 제대로 이루어지지 않았음을 나타낸다.


산점도에서도 위와 동일한 현상이 보였는데, 예측값이 실제값 주위에 고르게 분포되어 있지 않으며, 특정 값으로 몰리는 경향이 보인다. 

이를 통해 모델이 실제값과 예측값 사이의 관계를 제대로 학습하지 못하고 있으며, 편향된 예측을 하고 있음을 알 수 있다. 

모든 평가지표와 시각화 자료의 결과를 토대로 모델 성능이 현저히 낮다는 결과를 도출해낼 수 있었다.


---


## 5. 추후 개선 사항

모델의 실행 결과 처음 프로젝트를 시작했을때, 예상하였던 결과와 많이 달라져 있었다. model이 test data에 대해 잘 예측하지 못하고 있음을 알 수 있었다.

이러한 현상이 나타나는 이유에 대해서는 여러 원인이 있겠지만, 필자가 짐작하는 확실한 이유가 있다.

바로 data이다. 

필자는 train data가 가지고 있었던 문제를 해결하기 위해 직접 수학적 공식을 사용하여 문제를 해결하였다. 

하지만 수학 공식을 사용하면서 실수형이 모델 학습에 그대로 사용되게 되었고, 이는 정수형이었던 Test data와 단위 차이가 발생한 것이다. 그렇기에 이런 요소들이 모델을 학습하는데 영향을 주었고 그로인해 test의 예측이 낮아졌을거라 예상한다.

또한 범주형 열들을 encoding 할 때, 조건을 통해 진행했는데, 그 조건을 부여하는 방식이 틀렸기 때문에 data의 편향이나 사실적 오류가 발생했으리라 예상한다.

추후 test data에 대해 train data와 같이 직접 수학적 공식을 사용하여 단위를 맞춰주고 조건을 부여하는 방법을 다시 고안하여 학습 모델을 진행한 뒤 이전 결과와 어떤 차이가 있는지 비교할 예정이다.
