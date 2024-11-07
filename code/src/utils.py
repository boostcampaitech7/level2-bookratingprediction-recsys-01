import os
import time
import random
import numpy as np
import torch
import logging
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

def rmse(real: list, predict: list) -> float:
    '''
    [description]
    RMSE를 계산하는 함수입니다.

    [arguments]
    real : 실제 값입니다.
    predict : 예측 값입니다.

    [return]
    RMSE를 반환합니다.
    '''
    pred = np.array(predict)
    return np.sqrt(np.mean((real-pred) ** 2))

def train_valid_split(args, data):
    """
    Parameters
    ----------
    args.dataset.valid_ratio : float
        Train/Valid split 비율을 입력합니다.
    args.seed : int
        데이터 셔플 시 사용할 seed 값을 입력합니다.

    Returns
    -------
    data : dict
        data 내의 학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다.
    """
    if args.dataset.valid_ratio == 0:
        data['X_train'] = data['train'].drop('rating', axis=1)
        data['y_train'] = data['train']['rating']

    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
                                                            data['train'].drop(['rating'], axis=1),
                                                            data['train']['rating'],
                                                            test_size=args.dataset.valid_ratio,
                                                            random_state=args.seed,
                                                            shuffle=True
                                                            )
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    
    return data

class Setting:
    @staticmethod
    def seed_everything(seed):
        '''
        [description]
        seed 값을 고정시키는 함수입니다.

        [arguments]
        seed : seed 값
        '''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def __init__(self):
        now = time.localtime()
        now_date = time.strftime('%Y%m%d', now)
        now_hour = time.strftime('%X', now)
        save_time = now_date + '_' + now_hour.replace(':', '')
        self.save_time = save_time

    def get_log_path(self, args):
        '''
        [description]
        log file을 저장할 경로를 반환하는 함수입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

        [return]
        path : log file을 저장할 경로를 반환합니다.
        이 때, 경로는 saved/log/날짜_시간_모델명/ 입니다.
        '''
        path = os.path.join(args.train.log_dir, f'{self.save_time}_{args.model}/')
        self.make_dir(path)
        
        return path

    def get_submit_filename(self, args):
        '''
        [description]
        submit file을 저장할 경로를 반환하는 함수입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

        [return]
        filename : submit file을 저장할 경로를 반환합니다.
        이 때, 파일명은 submit/날짜_시간_모델명.csv 입니다.
        '''
        if args.predict == False:
            self.make_dir(args.train.submit_dir)
            filename = os.path.join(args.train.submit_dir, f'{self.save_time}_{args.model}.csv')
        else:
            filename = os.path.basename(args.checkpoint)
            filename = os.path.join(args.train.submit_dir, f'{filename}.csv')
            
        return filename

    def make_dir(self,path):
        '''
        [description]
        경로가 존재하지 않을 경우 해당 경로를 생성하며, 존재할 경우 pass를 하는 함수입니다.

        [arguments]
        path : 경로

        [return]
        path : 경로
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass
        return path


class Logger:
    def __init__(self, args, path):
        """
        [description]
        log file을 생성하는 클래스입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.
        path : log file을 저장할 경로를 전달받습니다.
        """
        self.args = args
        self.path = path

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('[%(asctime)s] - %(message)s')

        self.file_handler = logging.FileHandler(os.path.join(self.path, 'train.log'))
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def log(self, epoch, train_loss, valid_loss=None, valid_metrics=None):
        '''
        [description]
        log file에 epoch, train loss, valid loss를 기록하는 함수입니다.
        이 때, log file은 train.log로 저장됩니다.

        [arguments]
        epoch : epoch
        train_loss : train loss
        valid_loss : valid loss
        '''
        message = f'epoch : {epoch}/{self.args.train.epochs} | train loss : {train_loss:.3f}'
        if valid_loss:
            message += f' | valid loss : {valid_loss:.3f}'
        if valid_metrics:
            for metric, value in valid_metrics.items():
                message += f' | valid {metric.lower()} : {value:.3f}'
        self.logger.info(message)

    def close(self):
        '''
        [description]
        log file을 닫는 함수입니다.
        '''
        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()

    def save_args(self):
        '''
        [description]
        model에 사용된 args를 저장하는 함수입니다.
        이 때, 저장되는 파일명은 model.json으로 저장됩니다.
        '''
        with open(os.path.join(self.path, 'config.yaml'), 'w') as f:
            OmegaConf.save(self.args, f)

    def __del__(self):
        self.close()
    
