import argparse
import ast
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.decomposition import PCA
from catboost import CatBoostRegressor, cv, Pool
import sys
import sys
sys.path.append("./src")
from utils import Setting
from data.text_fixed_data import text_fixed_data_load, text_fixed_data_loader, text_fixed_data_split
from loss import loss as loss_module



class CatBoostRegression():
    def __init__(self, args, data):
        self.cbr = CatBoostRegressor(**args.model_args[args.model].params)
        self.data = data
        self.loss_fn = getattr(loss_module, args.loss)().to(args.device)

  
        if args.model_args[args.model].embed_module=='PCA':
            # PCA를 이용하여 16차원으로 축소
            pca = PCA(n_components=16)
            for df in [self.data['X_train'], self.data['X_valid'], self.data['test']]:
                for vector in ['user_summary_merge_vector', 'book_summary_vector']:
                    embeddings = pca.fit_transform(np.stack(df[vector]))
                    df[vector] = list(embeddings)
        else:
            # nn.Linear module 이용하여 16차원으로 축소
            linear = nn.Linear(args.model_args[args.model].word_dim, args.model_args[args.model].embed_dim)
            for df in [self.data['X_train'], self.data['X_valid'], self.data['test']]:
                for vector in ['user_summary_merge_vector', 'book_summary_vector']:
                    embeddings = linear(torch.Tensor(np.stack(df[vector])))
                    df[vector] = list(embeddings.detach())



    def fit(self, cat_features, embedding_features):
        self.cbr.fit(X=self.data['X_train'], y=self.data['y_train'], 
            cat_features=cat_features, embedding_features=embedding_features, 
            eval_set=(self.data['X_valid'], self.data['y_valid']))
    
        y_hat = self.cbr.predict(data=self.data['X_valid'])
        loss = self.loss_fn(self.data['y_valid'].float(), y_hat)

        msg = ''
        msg += f'\n\tValid RMSE Loss : {loss:.3f}'

        return msg
        

    def fit_all(self,cat_features, embedding_features):
        X_all = pd.concat([self.data['X_train'], self.data['X_valid']], axis=0)
        y_all = pd.concat([self.data['y_train'], self.data['y_valid']], axis=0)
        
        self.cbr = CatBoostRegressor(**args.model_args[args.model].params)
        self.cbr.fit(X_all, y_all,
                cat_features=cat_features, 
                embedding_features=embedding_features,
                verbose=False)
    
    
    def prediction(self):
        setting = Setting()

        preds = self.cbr.predict(data=self.data['test'])
        submission = pd.read_csv(self.args.dataset.data_path + 'sample_submission.csv')
        submission['rating'] = preds

        filename = setting.get_submit_filename(self.args)
        print(f'Save Predict: {filename}')
        submission.to_csv(filename, index=False)
        
        

if __name__ == '__main__':
    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    

    arg = parser.add_argument
    str2dict = lambda x: {k:int(v) for k,v in (i.split(':') for i in x.split(','))}

    # add basic arguments (no default value)
    arg('--config', '-c', '--c', type=str, 
        help='Configuration 파일을 설정합니다.', required=True)
    arg('--predict', '-p', '--p', '--pred', type=ast.literal_eval, 
        help='학습을 생략할지 여부를 설정할 수 있습니다.')
    arg('--checkpoint', '-ckpt', '--ckpt', type=str, 
        help='학습을 생략할 때 사용할 모델을 설정할 수 있습니다. 단, 하이퍼파라미터 세팅을 모두 정확하게 입력해야 합니다.')
    arg('--seed', '-s', '--s', type=int,
        help='데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.')
    arg('--device', '-d', '--d', type=str, 
        choices=['cuda', 'cpu', 'mps'], help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--wandb', '--w', '-w', type=ast.literal_eval, 
        help='wandb를 사용할지 여부를 설정할 수 있습니다.')
    arg('--wandb_project', '--wp', '-wp', type=str,
        help='wandb 프로젝트 이름을 설정할 수 있습니다.')
    arg('--run_name', '--rn', '-rn', '--r', '-r', type=str,
        help='wandb에서 사용할 run 이름을 설정할 수 있습니다.')
    arg('--model_args', '--ma', '-ma', type=ast.literal_eval)
    arg('--dataloader', '--dl', '-dl', type=ast.literal_eval)
    arg('--dataset', '--dset', '-dset', type=ast.literal_eval)
    arg('--optimizer', '-opt', '--opt', type=ast.literal_eval)
    arg('--loss', '-l', '--l', type=str)
    arg('--lr_scheduler', '-lr', '--lr', type=ast.literal_eval)
    arg('--metrics', '-met', '--met', type=ast.literal_eval)
    arg('--train', '-t', '--t', type=ast.literal_eval)

    
    args = parser.parse_args()


    ######################## Config with yaml
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]
    

    catboost = CatBoostRegression(args=config_yaml)
   
    # cat_feature 설정
    cat_features = ['user_id', 'isbn', 'age_category', 'country', 'state', 'city', 'age_country',
                    'book_title', 'book_author_preprocessing', 'isbn_country', 'isbn_book', 'isbn_publisher',
                    'publisher_preprocessing', 'language', 'category_preprocessing']
    embedding_features = ['user_summary_merge_vector', 'book_summary_vector']

    print("fitting") 
    
    # valid loss fitting을 확인하고 싶으면 아래의 코드로 실행 
    # loss_name = catboost.fit(cat_features=cat_features, embedding_features=embedding_features)
    # print(loss_name)
    
    # all data에 대해 fit한 뒤 inference 추출을 위해서는 아래 코드로 실행
    catboost.fit_all(cat_features=cat_features, embedding_features=embedding_features)
    
    print("prediction")
    catboost.predict()