<div align='center'>
<p align='center'>
    <img src="https://capsule-render.vercel.app/api?type=waving&color=ece700&height=250&section=header&text=Rec%20N%20Roll&fontSize=80&animation=fadeIn&fontAlignY=38&desc=Lv2%20Project&descAlignY=51&descAlign=80"/>
</p>
</div>

# 📚 LV.2 RecSys 프로젝트 : Book Rating Prediction


## 🏆 대회 소개
| 특징 | 설명 |
|:---:|---|
| 대회 주제 | 네이버 부스트캠프 AI-Tech 7기 RecSys level2 - RecSys 기초 프로젝트|
| 대회 설명 | 사용자의 책 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 예측하는 태스크입니다. |
| 데이터 구성 | `books.csv, users.csv, train_ratings.csv, test_ratings.csv` 총 네 개의 CSV 파일 |
| 평가 지표 | Root Mean Squared Error (RMSE)로 실제 평점과 예측 평점 간의 오차 측정 |

---
## 💻 팀 구성 및 역할
| 박재욱 | 서재은 | 임태우 | 최태순 | 허진경 |
|:---:|:---:|:---:|:---:|:---:|
|[<img src="https://github.com/user-attachments/assets/0c4ff6eb-95b0-4ee4-883c-b10c1a42be14" width=130>](https://github.com/park-jaeuk)|[<img src="https://github.com/user-attachments/assets/b6cff4bf-79c8-4946-896a-666dd54c63c7" width=130>](https://github.com/JaeEunSeo)|[<img src="https://github.com/user-attachments/assets/f6572f19-901b-4aea-b1c4-16a62a111e8d" width=130>](https://github.com/Cyberger)|[<img src="https://github.com/user-attachments/assets/a10088ec-29b4-47aa-bf6a-53520b6106ce" width=130>](https://github.com/choitaesoon)|[<img src="https://github.com/user-attachments/assets/7ab5112f-ca4b-4e54-a005-406756262384" width=130>](https://github.com/jinnk0)|
|NCF with CNN modeling & tuning|Feature Engineering, CatBoost with Text data modeling|NFM, NFFM modeling & tuning|EDA, Feature Engineering, FFM+DCN modeling|CVAE modeling & tuning|
---
## 📚 프로젝트 개요
|    개요    | 설명 |
|:---:| --- |
| 주제 | 해당 경진대회는 책 구매를 결정할 때 소비자에게 도움을 주기 위해 개인화된 책 추천 모델을 만드는 대회입니다. 책 한 권은 원고지 기준 800~1000매로 긴 분량이라 소비자들이 제목, 저자, 표지 등의 제한된 정보를 바탕으로 신중하게 선택해야 합니다. 주어진 데이터셋(`users.csv, books.csv, train_ratings.csv`)등을 활용해 각 사용자가 책에 부여할 평점을 예측하는 것이 목표입니다.  |
| 목표 | 주어지는 user, book, image, text 등의 데이터를 활용하여 평점을 예측하는 AI 알고리즘을 개발하는 것이 대회의 주된 목표입니다. |
| 평가 지표 | **Root Mean Squared Error (RMSE)**  |
| 개발 환경 | `GPU` : Tesla V100 Server 4대, `IDE` : VSCode, Jupyter Notebook, Google Colab |
| 협업 환경 | `Notion`(진행 상황 공유), `Github`(코드 및 데이터 공유), `Slack` , `카카오톡`(실시간 소통) |


### 데이터셋 구성
>- `books.csv` : 기본적인 책 정보

| 컬럼명 | 설명 |
| --- | --- |
|`isbn`|책을 구분하는 고유한 아이디|
|`book_title`|책의 제목|
|`book_author`|책을 집필한 작가|
|`year_of_publication`|책을 발행한 연도|
|`publisher`|책을 발행한 출판사|
|`img_url`|책 표지 이미지에 접속할 수 있는 url|
|`language`|출판 언어|
|`category`|책에 대한 카테고리|
|`summary`|책에 대한 요약|
|`img_path`|책 표지 이미지가 들어 있는 파일 경로|

>- `users.csv` : 사용자 정보

| 컬럼명 | 설명 |
| --- | --- |
|`user_id`|유저를 식별하는 아이디|
|`location`|유저의 위치 정보|
|`age`|유저의 나이|

>- `train_ratings.csv; test_ratings.csv` : train, test 정보

| 컬럼명 | 설명 |
| --- | --- |
|`user_id`|유저를 식별하는 고유 아이디|
|`isbn`|책을 식별하는 고유 아이디|
|`rating`|해당 유저가 해당 책에 대해 매긴 평점|




---
## 🕹️ 프로젝트 실행
### 디렉토리 구조

```
📦 level2-bookratingprediction-recsys-01
|-- config
|   |-- config.yaml
|   |-- sweep_CVAE.yaml
|-- data
|   |-- text_vector
|-- ensemble.py
|-- main.py
|-- requirement.txt
|-- src
    |-- __init__.py
    |-- data
    |   |-- __init__.py
    |   |-- all_data.py
    |   |-- context_image.py
    |   |-- handler
    |   |   |-- context_handling.py
    |   |   |-- image_handling.py
    |   |   |-- text_handling.py
    |   |-- text_context_data.py
    |   |-- text_data.py
    |-- ensembles
    |-- loss
    |   |-- loss.py
    |— models
    |   |— CVAE.py
    |   |— DCNwithFFM.py
    |   |— NCF.py
    |   |— __init__.py
    |   |— _helpers.py
    |— train
    |   |— __init__.py
    |   |— trainer.py
    |— utils.py
```

### Installation with pip
1. `pip install -r requirements.txt` 실행
2. Unzip train, dev, test csv files at /data directory
3. Upload sample_submission.csv at /data directory
```bash
# NCF with CNN
$ python main.py —config config/config.yaml —device cuda —m NCF —seed 42 —metrics "['RMSELoss']" -w True

# CVAE
$ python main.py  -c config/config.yaml  -m CVAE  -w True  -r CVAE_test

# NFM
$ python main.py  -c config/config.yaml  -m NFM  -w True 

# NFFM
python main.py  -c config/config.yaml  -m NFFM  -w True 

# DeepFFM
python main.py  -c config/config.yaml  -m DeepFFM  -w True

# CatBoost
python main.py -c config/config.yaml -m CatBoost -w True
```

