import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from .context_handling import BookProcessor, UserProcessor



class TextProcessor:
    def __init__(self, model_name : str, tokenizer, model, users, books, ratings, vector_create = False):
        """
        Parameters
        ----------
        model_name : str
            모델의 이름을 받습니다. (예: 'bert-base-uncased')
        tokenizer : Tokenizer
            텍스트 데이터를 모델에 입력하기 위한 토크나이저.
        model : Pre-trained Model
            텍스트 데이터를 벡터로 변환할 사전 학습된 모델.
        users : pd.DataFrame
            유저 정보에 대한 데이터프레임.
        books : pd.DataFrame
            책 정보에 대한 데이터프레임.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.users = users
        self.books = books
        self.ratings = ratings
        self.vector_create = vector_create

        try:
            self.model_name = model_name.split('/')[1]
        except IndexError:
            self.model_name = model_name


    def text_preprocessing(self, summary):
        """
        텍스트 데이터의 기본 전처리 (특수문자, 중복 공백 제거 등)
        """
        summary = re.sub("[^0-9a-zA-Z.,!?]", " ", summary)  # 특수문자 제거
        summary = re.sub("\s+", " ", summary)  # 중복 공백 제거
        return summary



    def text_to_vector(self, text):
        """
        텍스트를 모델을 사용하여 벡터로 변환합니다.
        """
        text_ = "[CLS] " + text + " [SEP]"
        tokenized = self.tokenizer.encode(text_, add_special_tokens=True)
        token_tensor = torch.tensor([tokenized], device=self.model.device)
        
        with torch.no_grad():
            outputs = self.model(token_tensor)
            sentence_embedding = outputs.pooler_output.squeeze(0)  # [CLS] 토큰의 임베딩만 사용
        
        return sentence_embedding.cpu().detach().numpy()



    def text_process_data(self, ratings, vector_create=False):
        """
        유저와 책 데이터에 대해 텍스트 데이터를 벡터화하고, 요약 정보를 병합합니다.
        """
        num2txt = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']
        books_ = self.books.copy()
        users_ = self.users.copy()
        nan_value = 'None'

        books_['summary'] = books_['summary'].fillna(nan_value)\
                                            .apply(self.text_preprocessing)\
                                            .replace({'': nan_value, ' ': nan_value})
        books_['summary_length'] = books_['summary'].apply(lambda x: len(x) if x is not None else 0)
        books_['review_count'] = books_['isbn'].map(ratings['isbn'].value_counts())
        users_['books_read'] = users_['user_id'].map(ratings.groupby('user_id')['isbn'].apply(list))

        if vector_create:
            # 벡터 생성 디렉토리 확인 및 생성
            if not os.path.exists('./data/text_vector'):
                os.makedirs('./data/text_vector')

            # 책의 요약 벡터 생성
            print('Create Item Summary Vector')
            book_summary_vector_list = []
            for title, summary in tqdm(zip(books_['book_title'], books_['summary']), total=len(books_)):
                prompt_ = f'Book Title: {title}\n Summary: {summary}\n'
                vector = self.text_to_vector(prompt_)
                book_summary_vector_list.append(vector)

            book_summary_vector_list = np.concatenate([books_['isbn'].values.reshape(-1, 1),
                                                      np.asarray(book_summary_vector_list, dtype=np.float32)], axis=1)

            np.save(f'./data/text_vector/book_summary_vector_{self.model_name}.npy', book_summary_vector_list)

            # 유저의 요약 벡터 생성
            print('Create User Summary Merge Vector')
            user_summary_merge_vector_list = []
            for books_read in tqdm(users_['books_read']):
                if not isinstance(books_read, list) and pd.isna(books_read):  # 유저가 읽은 책이 없는 경우
                    user_summary_merge_vector_list.append(np.zeros((768)))
                    continue
                
                read_books = books_[books_['isbn'].isin(books_read)][['book_title', 'summary', 'review_count']]
                read_books = read_books.sort_values('review_count', ascending=False).head(5)
                
                prompt_ = f'{num2txt[len(read_books)]} Books That You Read\n'
                for idx, (title, summary) in enumerate(zip(read_books['book_title'], read_books['summary'])):
                    summary = summary if len(summary) < 100 else f'{summary[:100]} ...'
                    prompt_ += f'{idx+1}. Book Title: {title}\n Summary: {summary}\n'
                vector = self.text_to_vector(prompt_)
                user_summary_merge_vector_list.append(vector)

            user_summary_merge_vector_list = np.concatenate([users_['user_id'].values.reshape(-1, 1),
                                                             np.asarray(user_summary_merge_vector_list, dtype=np.float32)], axis=1)

            np.save(f'./data/text_vector/user_summary_merge_vector_{self.model_name}.npy', user_summary_merge_vector_list)

        else:
            print('Check Vectorizer')
            print('Vector Load')
            book_summary_vector_list = np.load(f'./data/text_vector/book_summary_vector_{self.model_name}.npy', allow_pickle=True)
            user_summary_merge_vector_list = np.load(f'./data/text_vector/user_summary_merge_vector_{self.model_name}.npy', allow_pickle=True)

        # 결과 DataFrame 생성
        book_summary_vector_df = pd.DataFrame({'isbn': book_summary_vector_list[:, 0]})
        book_summary_vector_df['book_summary_vector'] = list(book_summary_vector_list[:, 1:].astype(np.float32))
        user_summary_vector_df = pd.DataFrame({'user_id': user_summary_merge_vector_list[:, 0]})
        user_summary_vector_df['user_summary_merge_vector'] = list(user_summary_merge_vector_list[:, 1:].astype(np.float32))

        return user_summary_vector_df, book_summary_vector_df
