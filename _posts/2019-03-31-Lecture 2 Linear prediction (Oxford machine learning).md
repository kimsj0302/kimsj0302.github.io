---
layout: post
title: "Lecture 2: Linear prediction (Oxford machine learning)"
date: 2019-03-31
categories: study
tags: [ML, Linear prediction,Oxford]
use_math: true
---

We are given a training dataset of $$n$$ instances of input-output pairs {$${x_{1:n},y_{1:n}}​$$}. 

Each input(predictors, covariates) $$x_i \in\mathbb{R}^{1 \times d}$$  is a vector with $$d​$$ attributes

The output(target) will be assumed to be univariate, $$y_i \in \mathbb{R}$$ 

Given this model and a new value of the input $$x_{n+1}$$, we can make prediction $$\hat{y}(x_{n+1})$$.


$$
x_{1:n}=\{x_1,x_2,x_3,x_4, \dots x_n\}
$$


이미지와 라벨과 같이 input은 라벨링이 되어있다. 그리고 $$d$$개의 attributes가 존재한다. 그들을 이용해서 우리는 하나의 실수 결과를 내는 것이 Linear prediction의 목표이다.

주어진 데이터 셋 {$${x_{1:n},y_{1:n}}$$}에 대해서 우리는 모델을 학습시켜 새로운 value input $$x_{n+1}$$에 대해서 예측값 $$\hat{y}(x_{n+1})$$을 얻는 것이다.



그것을 위해서 몇가지 단계가 필요하다. 

1. TRANING

   ​	먼저 데이터를 통해서 Learner를 이용해 model parameters를 학습해야 한다. 이 강좌에서는 model parameter를 $$\theta$$로 표현한다.

2. TESTING

   ​	$$x_{n+1},\theta$$를 이용해서 predictor를 통해 $$\hat{y}_{n+1}$$을 얻는 과정이다.

이제 간단한 모델로 부터 시작하다. 우리가 러닝을 하기 위해서는 objective function 또는 Energy function 또는 Loss function을 만들어야 한다. 러닝을 잘 했는지를 판단하기 위해서는 이 함수가 필요하다. 


$$
\hat{y}(x_i) = \theta_1 + x_i\theta_2
$$

$$
J(\theta)= \sum_{i=1}^{n}(y_i - \hat{y_i})^2 = \sum_{i=1}^{n} (y_i - \theta_1 - x_i\theta_2)^2
$$



우리는 2차원 평면에 점을 놓고 이 점에 맞는 선을 찾을 것이다.  그 선의 기울기는 $$\theta_2$$일 것이고 절편은 $$\theta_1$$이다.  그 식에서 각각의 차이(object function)를 계산 할 것이다. 이 모델은 대부분의 경우 잘 동작한다! 

(질문) 그렇게 되면 점과 직선의 거리를 구하지 않는다. 이게 최선인가?

(답) 점과 직선의 거리를 구하는 것이 효율은 좋다. 이에 대해서는 뒤에서 더 이야기 할 것이다. 우리는 데이터 특성에 맞는 모델을 선택해야 한다. 

## Linear prediction


$$
\hat{y_i}= \sum_{j=1}^{d} x_{ij}\theta_j=\require{cancel}\cancelto1{x_{i1}}\theta_1+x_{i2}\theta_2+\dots+x_{id}\theta_d
$$


usually $$x_{i1}=1$$. $$\theta_1​$$ is known as the bias or offset. 

### Matrix form

$$
\hat{\mathbf{y}}=X\theta
$$

with $$\hat{\mathbf{y}} \in \mathbb{R}^{n \times 1}, X \in \mathbb{R}^{n \times d}$$ and $$\theta \in \mathbb{R}^{d \times 1}​$$ 

### Quadratic Cost

Matrix form에서는 Cost function을 다음과 같이 정의 할 수 있다. 


$$
J(\theta)  = (\mathbf{y} - X\theta)^T(\mathbf{y}-X\theta) = \sum_{i=1
}^{n}(y_i-x_i^T\theta)^2
$$

## Optimization approach

우리의 목표는  아웃풋 라벨과 모델의 예측 간의 quadratic cost를 줄이는 것이다. 우리는 gradients를 이용할 것이다. 공을 굴리는 것을 생각하면 쉽다. gradients의 결과를 통해서 minimum을 찾으면 된다.

### Finding the solution by differentiation

Gradients를 구하기 위해서  $$J(\theta)= \sum_{i=1}^{n}(y_i - \hat{y_i})^2 = \sum_{i=1}^{n} (y_i - \theta_1 - x_i\theta_2)^2​$$ 이 식을 가지고 시작하면 


$$
\frac{\partial J(\theta)}{\partial \theta_1} \rightarrow g_1(\theta_1,\theta_2) \\
\frac{\partial J(\theta)}{\partial \theta_2} \rightarrow g_2(\theta_1,\theta_2)
$$


다음과 같은 식을 얻을 수 있을 것이다. 


$$
J(\theta)  = (\mathbf{y} - X\theta)^T(\mathbf{y}-X\theta)
$$


기본적인 선형 대수를 통해 우리는 matrix의 미분값이 다음과 같다는 것을 알 수 있다.


$$
\frac{\partial A\theta}{\partial \theta} = A^T \\
\frac{\partial \theta^TA\theta}{\partial \theta}= 2A^T\theta
$$


이를 통해 미분 값을 실제로 구해보면


$$
\begin{align}
\frac{\partial J(\theta)}{\partial \theta} &= \frac{\partial}{\partial \theta}(y^ty - 2y^TX\theta+\theta^TX^TX\theta)\\&= 0 - 2X^Ty + 2X^TX\theta \overset{equal}{=} 0
\end{align}
$$


미분값이 0이 되는 경우가 최소일 것이기 때문에


$$
\theta =(X^TX)^{-1}y^TX
$$

---

[Course Link](<https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/>)

이 글은 Oxford machine learning을 듣고 요약한 글 입니다. 

