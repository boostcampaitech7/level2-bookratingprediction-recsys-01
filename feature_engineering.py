import pandas as pd
import re
import numpy as np


 #### book feature engineering   


class BookFeatureEngineering:
    def __init__(self, book_df):
        self.df = book_df

    def text_preprocessing(self, summary):
        """
        Parameters
        ----------
        summary : str
            text 관련 기본적인 전처리를 하기 위한 텍스트 데이터를 입력합니다.
        ----------
        """
        summary = re.sub("[.,\'\"!?]", "", summary)  # 불필요한 기호 제거
        summary = re.sub("[^0-9a-zA-Z\\s]", " ", summary)  # 알파벳과 숫자 및 공백 제외
        summary = re.sub("\\s+", " ", summary)  # 여러 공백을 하나로
        summary = summary.lower()  # 소문자로 변환
        return summary
    
    
    
    def book_preprocess(self, df):
        # author가 없는 row가 1개이므로 해당 행은 제거
        df.drop(index=df.loc[df['book_author'].isna()].index, inplace=True)
        
        # language를 모드로 대체
        df['language'] = df['language'].fillna(
            df.groupby('isbn_country')['language'].transform(
                lambda x: x.mode()[0] if not x.mode().empty else 'en'
            )
        )
        
        # text_preprocessing 적용
        df["book_title_preprocessing"] = df["book_title"].apply(lambda x: self.text_preprocessing(x))
        df["book_author_preprocessing"] = df["book_author"].apply(lambda x: self.text_preprocessing(x))
        df["publisher_preprocessing"] = df["publisher"].apply(lambda x: self.text_preprocessing(x))
        
        # cateory preprocessing(5개 이하인 항목은 others)
        
        df['category_preprocessing'] = df['category'].apply(lambda x: re.sub('[\W_]+', ' ', str(x).lower()).strip())

        category_counts = df['category_preprocessing'].value_counts()
        df['category_preprocessing'] = df['category_preprocessing'].apply(lambda x: x if category_counts[x] >= 5 else 'others')
        df["category_preprocessing"] = df["category_preprocessing"].apply(lambda x: self.text_preprocessing(x))

        return df


    def create_features_isbn(self, df):
        df['isbn_country'] = df["isbn"].apply(lambda x: x[:1])
        df['isbn_publisher'] = df["isbn"].apply(lambda x: x[1:6])
        df['isbn_book'] = df["isbn"].apply(lambda x: x[6:9])
        df['isbn_check'] = df["isbn"].apply(lambda x: x[-1])
        return df
    
    def create_features_years(self, df):
        def preprocess_year(x, weighted_average):
            if x <= 1970:
                return 1970
            elif (x > 1970) and (x <= 1980):
                return 1980
            elif (x > 1980) and (x <= 1985):
                return 1985
            elif (x > 1985) and (x <= 1990):
                return 1990
            elif (x > 1990) and (x <= 1995):
                return 1995
            elif (x > 1995) and (x <= 2000):
                return 2000
            else:
                return weighted_average  # 2001년 이후의 가중 평균값

        # 2001년 이후의 연도들에 대한 가중 평균 계산
        after_2000 = df[df['year_of_publication'] > 2000]
        weights = after_2000['year_of_publication'].value_counts()
        weighted_average = ((weights.index * weights).sum() / weights.sum()).round(0)

        # preprocess_year를 적용하여 years 컬럼 생성
        df['years'] = df['year_of_publication'].apply(lambda x: preprocess_year(x, weighted_average))

        return df
    
    
    def final_preprocess(self):
        self.df = self.create_features_isbn(self.df)  # ISBN 관련 feature 생성
        self.df = self.create_features_years(self.df)  # 연도 관련 feature 생성
        self.df = self.book_preprocess(self.df)
        return self.df  # 최종 데이터프레임 반환

    
    
 ####### user feature engineering   

class UserFeatureEngineering:
    def __init__(self, df):
        self.df = df

    def user_preprocess(self, df):
        # 특수한 경우 제거(eda를 통해 확인함 -> location 정보가 이상한 부분)
        df.loc[21896, ['city', 'state', 'country']] = ['standrews', 'guernsey', 'channelislands']
        df.loc[28093, ['city', 'state', 'country']] = ['standrews', 'guernsey', 'channelislands']
        df.loc[64797, ['city', 'state', 'country']] = ['standrews', 'guernsey', 'channelislands']

        
        def fill_missing_values(df):
            previous_counts = None

            while True:
                # 1. city만 공백인 경우
                city_nan_only = df[df['country'].notna() & df['state'].notna() & df['city'].isna()].shape[0]
                # 2. state만 공백인 경우
                state_nan_only = df[df['country'].notna() & df['state'].isna() & df['city'].notna()].shape[0]
                # 3. country만 공백인 경우
                country_nan_only = df[df['country'].isna() & df['state'].notna() & df['city'].notna()].shape[0]
                # 4. city와 state가 공백인 경우
                city_state_nan_only = df[df['country'].notna() & df['state'].isna() & df['city'].isna()].shape[0]
                # 5. city와 country가 공백인 경우
                city_country_nan_only = df[df['country'].isna() & df['state'].notna() & df['city'].isna()].shape[0]
                # 6. state와 country가 공백인 경우
                state_country_nan_only = df[df['country'].isna() & df['state'].isna() & df['city'].notna()].shape[0]
                # 7. 전부 공백인 경우
                all_nan_only = df[df['country'].isna() & df['state'].isna() & df['city'].isna()].shape[0]

                # 현재 카운트를 리스트로 저장
                current_counts = [city_nan_only, state_nan_only, country_nan_only,
                                city_state_nan_only, city_country_nan_only,
                                state_country_nan_only, all_nan_only]

                # 이전 카운트와 비교
                if previous_counts is not None and current_counts == previous_counts:
                    break

                # 이전 카운트를 현재 카운트로 업데이트
                previous_counts = current_counts

                # 1. city만 공백인 경우 결측치 대체
                # country와 state의 group에 따른 최빈값으로 대체
                city_df = df[df['country'].notna() & df['state'].notna()]
                city_df['city'] = city_df['city'].fillna(
                    city_df.groupby(['country','state'])['city'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                )
                city_df['city'] = city_df['city'].fillna(
                    city_df.groupby(['country'])['city'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan) 
                )
                city_df['city'] = city_df['city'].fillna(
                    city_df.groupby(['state'])['city'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'unknown')
                )
                # 2. state만 공백인 경우 결측치 대체
                # country와 city의 group에 따른 최빈값으로 대체
                state_df = df[df['country'].notna() & df['city'].notna()]
                state_df['state'] = state_df['state'].fillna(
                    state_df.groupby(['country','city'])['state'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan) 
                )
                state_df = df[df['country'].notna() & df['city'].notna()]
                state_df['state'] = state_df['state'].fillna(
                    state_df.groupby(['country'])['state'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                )
                state_df = df[df['country'].notna() & df['city'].notna()]
                state_df['state'] = state_df['state'].fillna(
                    state_df.groupby(['city'])['state'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'unknown')
                )
                # 3. country만 공백인 경우 결측치 대체
                # state와 city의 group에 따른 최빈값으로 대체
                coutry_df = df[df['state'].notna() & df['city'].notna()]
                coutry_df['country'] = coutry_df['country'].fillna(
                    coutry_df.groupby(['state','city'])['country'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                )
                coutry_df['country'] = coutry_df['country'].fillna(
                    coutry_df.groupby('state')['country'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                )
                coutry_df['country'] = coutry_df['country'].fillna(
                    coutry_df.groupby('city')['country'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'unknown')
                )

                # 해당 값들로 update 실시
                df.update(city_df[['city']])
                df.update(state_df[['state']])
                df.update(coutry_df[['country']])
                # 4. city와 state가 공백인 경우 결측치 대체
                # country group에 따른 최빈값으로 대체
                city_state_df = df[df['country'].notna()]
                city_state_df['state'] = city_state_df['state'].fillna(
                    city_state_df.groupby('country')['state'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan) 
                )
                city_state_df['city'] = city_state_df['city'].fillna(
                    city_state_df.groupby('country')['city'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'unknown')
                )

                # 5. city와 country가 공백인 경우 결측치 대체
                # state group에 따른 최빈값으로 대체
                city_country_df = df[df['state'].notna()]
                city_country_df['country'] = city_country_df['country'].fillna(
                    city_country_df.groupby('state')['country'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan) 
                )
                city_country_df['city'] = city_country_df['city'].fillna(
                    city_country_df.groupby('state')['city'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'unknown')
                )
                # 6. state와 country가 공백인 경우 결측치 대체
                # city group에 따른 최빈값으로 대체
                state_country_df = df[df['city'].notna()]
                state_country_df['country'] = state_country_df['country'].fillna(
                    state_country_df.groupby('city')['country'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan) 
                )
                state_country_df['state'] = state_country_df['state'].fillna(
                    state_country_df.groupby('city')['state'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'unknown')
                )
                # 해당 값들로 update 실시
                df.update(city_state_df['state'])
                df.update(city_state_df['city'])

                df.update(city_country_df['state'])
                df.update(city_country_df['city'] )

                df.update(state_country_df[['state']])
                df.update(state_country_df[['country']])
                # 7. 모두 공백인 경우 결측치 대체
                # 각 column의 최빈값으로 대체
                for column in ['city','state','country']:
                    mode_value = df[column].mode()[0]
                    df[column].fillna(mode_value, inplace=True)
                    
                df.replace('unknown', np.nan, inplace=True)
            
            # 반복 이후 대체
            
            # city만 없는 경우
            city_nan_only = df[df['country'].notna() & df['state'].notna() & df['city'].isna()]
            city_nan_only['city'].fillna(city_nan_only['state'], inplace=True)
            df.update(city_nan_only['city'])

            # state만 없는 경우
            state_nan_only = df[df['country'].notna() & df['state'].isna() & df['city'].notna()]
            state_nan_only['state'].fillna(state_nan_only['city'], inplace=True)
            df.update(state_nan_only['state'])

            # country만 없는 경우
            country_nan_only = df[df['country'].isna() & df['state'].notna() & df['city'].notna()]
            country_nan_only['country'].fillna(country_nan_only['state'], inplace=True)
            df.update(country_nan_only['country'])

            # 4. city와 state가 공백인 경우
            city_state_nan_only = df[df['country'].notna() & df['state'].isna() & df['city'].isna()]
            city_state_nan_only['city'].fillna(city_state_nan_only['country'], inplace=True)
            city_state_nan_only['state'].fillna(city_state_nan_only['country'], inplace=True)
            df.update(city_state_nan_only['city'])
            df.update(city_state_nan_only['state'])

            # 5. city와 country가 공백인 경우
            city_country_nan_only = df[df['country'].isna() & df['state'].notna() & df['city'].isna()]
            city_country_nan_only['city'].fillna(city_country_nan_only['state'], inplace=True)
            city_country_nan_only['country'].fillna(city_country_nan_only['state'], inplace=True)
            df.update(city_country_nan_only['city'])
            df.update(city_country_nan_only['country'])

            # 6. state와 country가 공백인 경우
            state_country_nan_only = df[df['country'].isna() & df['state'].isna() & df['city'].notna()]
            state_country_nan_only['state'].fillna(state_country_nan_only['city'], inplace=True)
            state_country_nan_only['country'].fillna(state_country_nan_only['city'], inplace=True)
            df.update(state_country_nan_only['state'])
            df.update(state_country_nan_only['country'])
            
            
            # 7. 모두 공백인 경우 결측치 대체
            # 각 column의 최빈값으로 대체
            all_notnon_only = df[df['country'].notna() & df['state'].notna() & df['city'].notna()]
            most_common_combo = (
                all_notnon_only.groupby(['country', 'state', 'city'])
                .size()
                .reset_index(name='count')
                .sort_values(by='count', ascending=False)
                .head(1)
            )
            common_country = most_common_combo['country'].values[0]
            common_state = most_common_combo['state'].values[0]
            common_city = most_common_combo['city'].values[0]

            df['country'].fillna(common_country, inplace=True)
            df['state'].fillna(common_state, inplace=True)
            df['city'].fillna(common_city, inplace=True)
            
            return df
 
        df = fill_missing_values(df)

        # age 결측치 대체
        df['age'] = df.groupby(['country', 'state', 'city'])['age'].transform(lambda x: x.fillna(x.mean().round(0) if not pd.isna(x.mean()) else x))
        df['age'] = df.groupby(['country', 'state'])['age'].transform(lambda x: x.fillna(x.mean().round(0) if not pd.isna(x.mean()) else x))
        df['age'] = df.groupby('country')['age'].transform(lambda x: x.fillna(x.mean().round(0) if not pd.isna(x.mean()) else x))
        df['age'].fillna(df['age'].mean().round(0), inplace=True)


        return df

    def create_features_location(self, df):
        df['location'] = df['location'].str.replace(r'[^a-zA-Z:,]', '', regex = True)

        df['city'] = df['location'].apply(lambda x : x.split(',')[0].strip())
        df['state'] = df['location'].apply(lambda x : x.split(',')[1].strip())
        df['country'] = df['location'].apply(lambda x : x.split(',')[2].strip())

        df = df.replace('na', np.nan)
        df = df.replace('n/a', np.nan)
        df = df.replace('', np.nan)
        
        return df
    
    
    def create_features_age(self, df):
        age_bins = [0, 5, 12, 15, 18, 22, 26, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, np.inf]
        age_labels = [
            "0-5세", "6-12세", "13-15세", "16-18세", "19-22세", "23-26세", "27-30세", 
            "31-35세", "36-40세", "41-45세", "46-50세", "51-55세", "56-60세", 
            "61-65세", "66-70세", "71-75세", "76세 이상"
        ]

        df['age_category'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
        # Calculate the weighted average age for individuals in the "76세 이상" category
        over_76_avg = df.loc[df['age'] >= 76, 'age'].mean()
        over_76_avg_filled = over_76_avg if not np.isnan(over_76_avg) else 76  # Default to 76 if NaN

        df.loc[df['age_category'] == "76세 이상", 'age'] = over_76_avg_filled
        
        return df   
    

    def final_preprocess(self):
        self.df = self.create_features_location(self.df)  # location 관련 feature 생성
        self.df = self.user_preprocess(self.df)
        self.df = self.create_features_age(self.df) # age 관련 feature 생성
        
        return self.df  # 최종 데이터프레임 반환
