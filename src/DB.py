# -*- coding: utf-8 -*-

import os

import pandas as pd
from tqdm import tqdm

DB_dir = '../database'
DB_csv = '../data.csv'


class Database(object):

    def __init__(self, image=None):
        if image is None:
            self._gen_csv()
            self.data = pd.read_csv(DB_csv)
            self.classes = set(self.data["cls"])

    def _gen_csv(self):
        if os.path.exists(DB_csv):
            return
        with open(DB_csv, 'w', encoding='UTF-8') as f:
            f.write("img,cls")
            for root, _, files in tqdm(os.walk(DB_dir, topdown=False), desc='Creating Database'):
                cls = root.split('/')[-1]
                for name in files:
                    if not name.endswith('.jpg'):
                        continue
                    img = os.path.join(root, name)
                    f.write("\n{},{}".format(img, cls))

    def __len__(self):
        return len(self.data)

    def get_class(self):
        return self.classes

    def get_data(self):
        return self.data

    def __add_data(self, img_path, img_cls=None):
        """增，去存未定的方法，请勿使用"""
        if os.path.isdir(img_path):
            print(True)
        else:
            path_cls = os.path.join(DB_dir, img_cls)
            img = os.path.join(path_cls, os.path.split(img_path)[-1])
            if os.path.exists(path_cls):
                raise Exception('There exists a file with the same name')

            print(path_cls,img)
            # with open(DB_csv, 'a', encoding='UTF-8') as f:
            #     cls = path_cls.split('/')[-1]
            #     img = os.path.join(path_cls, os.path.split(img_path)[-1])
            #     f.write("\n{},{}".format(img, cls))

        for t in tqdm(os.walk(img_path, topdown=False), desc='Creating Database'):
            print(t)
    def __del_data(self):
        """删，去存未定的方法，请勿使用"""
        pass

class OneData(Database):
    '''定义了一个新类，继承Database，用于实现单个图片的单独输入。'''

    def __init__(self, image: str):
        super().__init__(image)
        self.data = pd.DataFrame({'cls': 'temporary', 'img': image}, index=[0])
        self.classes = set(self.data["cls"])

    def _gen_csv(self):
        raise Exception("class OneData isn't have _gen_csv method.")


if __name__ == "__main__":
    db = Database()
    data = db.get_data()
    classes = db.get_class()

    # print("DB length:", len(db))
    # print(classes)


    # print(data)
    db.add_data(r'F:\CBIR_pochih_CBIR\CBIR2\src\tests\10.jpg', 'tests')

