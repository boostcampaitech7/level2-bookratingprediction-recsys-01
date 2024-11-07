import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from .handler.context_handling import BookProcessor, UserProcessor
from .handler.text_handling import TextProcessor


class Text_Dataset(Dataset):
    def __init__(self, user_book_vector, user_summary_vector, book_summary_vector, rating=None):
        """
        Parameters
        ----------
        user_book_vector : np.ndarray
            벡터화된 유저와 책 데이터를 입력합니다.
        user_summary_vector : np.ndarray
            벡터화된 유저에 대한 요약 정보 데이터를 입력합니다.
        book_summary_vector : np.ndarray
            벡터화된 책에 대한 요약 정보 데이터 입력합니다.
        label : np.ndarray
            정답 데이터를 입력합니다.
        ----------
        """
        self.user_book_vector = user_book_vector
        self.user_summary_vector = user_summary_vector
        self.book_summary_vector = book_summary_vector
        self.rating = rating
    def __len__(self):
        return self.user_book_vector.shape[0]
    def __getitem__(self, i):
        return {
                'user_book_vector' : torch.tensor(self.user_book_vector[i], dtype=torch.long),
                'user_summary_vector' : torch.tensor(self.user_summary_vector[i], dtype=torch.float32),
                'book_summary_vector' : torch.tensor(self.book_summary_vector[i], dtype=torch.float32),
                'rating' : torch.tensor(self.rating[i], dtype=torch.float32),
                } if self.rating is not None else \
                {
                'user_book_vector' : torch.tensor(self.user_book_vector[i], dtype=torch.long),
                'user_summary_vector' : torch.tensor(self.user_summary_vector[i], dtype=torch.float32),
                'book_summary_vector' : torch.tensor(self.book_summary_vector[i], dtype=torch.float32),
                }


def process_text_data(args, ratings, users, books, tokenizer, model, model_name):
    text_processor = TextProcessor(model_name, tokenizer, model, users, books, ratings, args.model_args[args.model].vector_create)
    user_vectors, book_vectors = text_processor.text_process_data(ratings, args.model_args[args.model].vector_create)

    return user_vectors, book_vectors

def text_data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser
    args.model_args[args.model].pretrained_model : str
        사전학습된 모델을 설정할 수 있는 parser
    args.model_args[args.model].vector_create : bool
        텍스트 데이터 벡터화 및 저장 여부를 설정할 수 있는 parser
        False로 설정하면 기존에 저장된 벡터를 불러옵니다.

    Returns
    -------
    data : dict
        학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다.
    """
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    tokenizer = AutoTokenizer.from_pretrained(args.model_args[args.model].pretrained_model)
    model = AutoModel.from_pretrained(args.model_args[args.model].pretrained_model).to(device=args.device)
    model.eval()
    users_summary_vec, books_summary_vec = process_text_data(args, train, users, books, tokenizer, model, args.model_args[args.model].pretrained_model)

    #--------------------------------------------------------------------------------------------------------------------------------------#
    print("Context processing")
    
    # feature engineering
    book_df = books.copy()
    user_df = users.copy()
    
    book_fe = BookProcessor(book_df)
    book_df = book_fe.final_process()

    user_fe = UserProcessor(user_df)
    user_df = user_fe.final_process()

    book_df = pd.merge(book_df, books_summary_vec, on='isbn', how='left')
    user_df = pd.merge(user_df, users_summary_vec, on='user_id', how='left')  

    #--------------------------------------------------------------------------------------------------------------------------------------#

    # 유저 및 책 정보를 합쳐서 데이터 프레임 생성
    # 사용할 컬럼을 user_features와 book_features에 정의 (단, 모두 범주형 데이터로 가정)
    user_features = ['user_id', 'age_category', 'country', 'state', 'city', 'age_country']
    book_features = ['isbn', 'book_title', 'book_author_preprocessing', 'isbn_country', 'isbn_book', 'isbn_publisher','publisher_preprocessing', 'language', 'category_preprocessing']
    sparse_cols = ['user_id', 'isbn'] + list(set(user_features + book_features) - {'user_id', 'isbn'}) if args.model == 'NCF' \
                   else user_features + book_features

    # 선택한 컬럼만 추출하여 데이터 조인
    train_df = train.merge(user_df, on='user_id', how='left')\
                    .merge(book_df, on='isbn', how='left')[sparse_cols + ['user_summary_merge_vector', 'book_summary_vector','rating']]
    train_df = train_df.drop(index=train_df.loc[train_df['book_author_preprocessing'].isna()].index)    
    test_df = test.merge(user_df, on='user_id', how='left')\
                  .merge(book_df, on='isbn', how='left')[sparse_cols + ['user_summary_merge_vector', 'book_summary_vector']]
    all_df = pd.concat([train_df, test_df], axis=0)

    # feature_cols의 데이터만 라벨 인코딩하고 인덱스 정보를 저장
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown')
        train_df[col] = train_df[col].fillna('unknown')
        test_df[col] = test_df[col].fillna('unknown')
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        train_df[col] = pd.Categorical(train_df[col], categories=unique_labels).codes
        test_df[col] = pd.Categorical(test_df[col], categories=unique_labels).codes
    
    field_dims = [len(label2idx[col]) for col in sparse_cols]


    data = {
            'train':train_df,
            'test':test_df,
            'field_names':sparse_cols,
            'field_dims':field_dims,
            'label2idx':label2idx,
            'idx2label':idx2label,
            'sub':sub,
            }
    
    return data



def text_data_loader(args, data):
    """
    Parameters
    ----------
    args.dataloader.batch_size : int
        데이터 batch에 사용할 데이터 사이즈
    args.dataloader.shuffle : bool
        data shuffle 여부
    args.dataloader.num_workers: int
        dataloader에서 사용할 멀티프로세서 수
    args.dataset.valid_ratio : float
        Train/Valid split 비율로, 0일 경우에 대한 처리를 위해 사용
    data : dict
        text_data_load()에서 반환된 데이터

    Returns
    -------
    data : dict
        Text_Dataset 형태의 학습/검증/테스트 데이터를 DataLoader로 변환하여 추가한 후 반환합니다.
    """
    train_dataset = Text_Dataset(
                                data['X_train'][data['field_names']].values,
                                data['X_train']['user_summary_merge_vector'].values,
                                data['X_train']['book_summary_vector'].values,
                                data['y_train'].values
                                )
    valid_dataset = Text_Dataset(
                                data['X_valid'][data['field_names']].values,
                                data['X_valid']['user_summary_merge_vector'].values,
                                data['X_valid']['book_summary_vector'].values,
                                data['y_valid'].values
                                ) if args.dataset.valid_ratio != 0 else None
    test_dataset = Text_Dataset(
                                data['test'][data['field_names']].values,
                                data['test']['user_summary_merge_vector'].values,
                                data['test']['book_summary_vector'].values,
                                )
    if args.ML:
        return data

    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers) if args.dataset.valid_ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers)
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
    
    return data
