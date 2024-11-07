from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.decomposition import PCA
from catboost import CatBoostRegressor
from utils import Setting
from loss import loss as loss_module



class CatBoostRegression():
    def __init__(self, args, data):
        self.args = args
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
        
        self.cbr = CatBoostRegressor(**self.args.model_args[self.args.model].params)
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
        