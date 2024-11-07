import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2

class ImageProcessor:
    def __init__(self, books, img_size):
        """
        Parameters
        ----------
        books : pd.DataFrame
            책 정보에 대한 데이터프레임.
        img_size : int
            이미지 크기
        """
        self.books = books
        self.img_size = img_size

    def image_vector(self, path):
        """
        Parameters
        ----------
        path : str
            이미지가 존재하는 경로를 입력합니다.

        Returns
        -------
        img_fe : np.ndarray
            이미지를 벡터화한 결과를 반환합니다.
            베이스라인에서는 grayscale일 경우 RGB로 변경한 뒤, img_size x img_size 로 사이즈를 맞추어 numpy로 반환합니다.
        """
        img = Image.open(path)
        transform = v2.Compose([
            v2.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
            v2.Resize((self.img_size, self.img_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transform(img).numpy()
    
    def process_img_data(self):
        """
        Parameters
        ----------
        books : pd.DataFrame
            책 정보에 대한 데이터 프레임을 입력합니다.
        
        Returns
        -------
        books_ : pd.DataFrame
            이미지 정보를 벡터화하여 추가한 데이터 프레임을 반환합니다.
        """
        books_ = self.books.copy()
        books_['img_path'] = books_['img_path'].apply(lambda x: f'../data/{x}')
        img_vecs = []
        for idx in tqdm(books_.index):
            img_vec = self.image_vector(books_.loc[idx, 'img_path'])
            img_vecs.append(img_vec)

        books_['img_vector'] = img_vecs

        return books_