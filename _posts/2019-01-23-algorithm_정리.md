---
layout: post
title: "알고리즘 정리"
date: 2019-01-23
categories: study
tags: [ML, jekyll]
use_math: true
---
# 목차

1.  [Apriori 알고리즘](#apriori_algorithm)<br />
1.1.  [개요](#summary)<br />
1.2. [규칙과 규칙의 효용성](#rule)<br />
1.3. [규칙 생성](#generate_rule)<br />
1.4. [Code](#apriori_code)

# Apriori 알고리즘 <a class="apriori" id="apriori_algorithm"></a>

https://ratsgo.github.io/machine%20learning/2017/04/08/apriori/ 를 참고하여 다시 정리한 글임

<br />


### 개요 <a class="apriori" id="summary"></a>
***

 데이터들에 대한 발생 빈도를 기반으로 각 데이터 간의 연관관계를 밝히기 위한 방법이다. 구현이 간단하며 성능 또한 좋다. 패턴 분석에 많이 사용된다. contents-based recommendation의 기본이다.

 먼저 용어를 정리해보자. 조건절(Antecedent)은 '~라면'에 해당하고, 결과절(Consquent)은 '~한다', 아이템 집합(Item set)은 각각의 절에 들어가는 아이템들의 집합을 의미한다. 예를 들어 다음과 같은 문장을 생각해보자. "달걀을 구매하는 사람들은 라면도 산다" 여기서 조건절은 달걀을 구매 , 결과절은 라면 구매, 아이템 집합은 각각 '달걀'과 '라면'이 될 것이다. 2개의 아이템 집합은 서로 *mutually exclusive* 여야 하는데 이는 조건절과 결과절에 교집합이 존재하면 안된다는 것이다. 


<br />

### 규칙과 규칙의 효용성 <a class="apriori" id="rule"></a>
***
 우리는 조건절과 결과절에 아이템을 집어 넣어 규칙을 만들 수 있다. 'A를 사는 사람은 B도 산다'와 같이 말이다. 그렇다면 이를 이제 어떻게 평가해야 할까? 규칙이 좋은지 나타내는 지표는 **지지도(support)**와 **신뢰도(confidence)**,**향상도(lift)** 가 있습니다. 
\begin{align}
\text{For the rule} \space A \to B
\end{align}
1. 지지도 support는 조건절이 일어날 확률을 말한다. 

\begin{align}
supprot(A) = P(A)
\end{align}

2. 신뢰도 confidence는 조건절이 주어졌을 때, 결과절이 일어날 확률로 정의됨

\begin{align}
confidence(A \to B) = \frac{P(A|B)}{P(A)}
\end{align}

3. 향상도 lift는 아래와 같이 조건절과 결과절이 서로 독립일 때와 비교해 얼마나 두 사건이 동시에 발생하는지에 대한 비율로 나타낸다. (1일 경우 조건절과 결과절이 독립 : 연관성 없음)

\begin{align}
lift(A \to B)= \frac{P(A|B)}{P(A) \cdot P(B)}
\end{align}

이 세가지 지표가 모두 커야만 효과적인 규칙이다.


<br />

### 규칙 생성 <a class="apriori" id="generate_rule"></a>
***
아이템이 $n$개 일때 탐색을 해야하는 모든 경우의 수는 $n(n-1)$이다. **빈발 집합(frequent itemset)** 이란 정해진 최소 지지도 이상을 자기는 항목집합을 말한다. 우리가 원하는 것은 빈발 집합이기 때문에 이를 효율적으로 구하기 위해 Apriori algorithm을 사용한다. 

아이템 집합 ${A}$의 지지도 $P(A)$가 0.1이라고 하자. 그렇다면 A를 포함하는 아이템 집합들의 지지도는 0.1을 넘지는 못할 것이다. 때문에 최소지지도를 만족하지 못하는 아이템 집합을 발견하면 그것의 초월 집합(집합의 원소를 부분집합으로 가지는 다른 집합, 위에서는 ${A}$는 ${A,B}$,${A,B,C}$의 **초월 집합**)은 계산을 하지 않아도 된다.
<img src="https://i.imgur.com/tncW2Gn.png">

<br />

#### Sample Code<a class="apriori" id="apriori_code"></a>
***
 **https://github.com/jjkyun/DataMining/tree/master/apriori**
 
 ### 문제점
 ***
 Apriori 알고리즘은 느리다.
 때문에 이를 보안하기 위한 3가지 방법이 있다.
 

# Frequent Patterns Growth (FP-Growth) 알고리즘
***

**Apriori algorithm** 과 같은 결과를 내는 알고리즘이다. 하지만 전체 DB를 2번만 보기 때문에 시간이 매우 빠르다는 장점이 있다. 2개의 step으로 진행된다. step 1에서는 **FP-tree**를 구성한다. step 2에서는 FP-tree를 돌면서 결과를 얻는다.

### step 1 : FP-tree 구성
***
https://chih-ling-hsu.github.io/2017/03/25/frequent-itemset-generation-using-fp-growth#mining-tree
<img src="https://i.imgur.com/nty7dVx.png">

#### Make Transaction Database
먼저 Transcation Database를 만든다. 
1. 1-itemset을 구하고 최소 지지도 이하를 가지는 것을 제거한다.
2. count가 높은 순서대로 DB의 각 라인을 정렬하여 Transaction DB를 만든다.

#### Make FP-Tree
1. Root Node를 만든다.(NULL)
2. Transaction DB에 있는 각각의 데이터마다 FP-tree에 집어 넣는다. 집어 넣을 때에는 위에서 구한 순서에 따라서 높은 count를 가지는 item이 root node에 가깝도록 집어 넣는다. 이미 기존에 있는 FP-tree에 있는 item을 넣을 때에는 그 item의 발생 수를 1 증가 시킨다. 새로운 item이면 node를 leaf node에 추가로 만들어주고, {item}:1과 같은 형태로 발생 수를 초기화 한다.  
3. 새로 node가 생성되었다면 기존의 같은 아이템을 가지고 있는 node와 linking 해준다.
4. 2~3을 모든 Transaction에 대해서 반복한다.

<img src="https://i.imgur.com/QXBcLWn.png">

#### Mining Tree
1. 1-itemset에서 count가 가장 낮은 item부터 시작한다. item을 하나 선택한다.
2. 선택한 아이템으로 끝나는(모든 leaf node가 선택한 item인) subtree를 찾는다. 그리고 그 item을 제거한다. 이는 그 item을 포함하는 frquent itemset을 만드는 것이다. 
3. 두번째 과정에서 다시 1번과 같은 진행을 한다. 그렇다면 재귀적으로 모든 frequent itemset을 만들 수 있다.


