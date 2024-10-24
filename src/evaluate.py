# -*- coding: utf-8 -*-

import numpy as np
from scipy import spatial
from tqdm import tqdm
import pickle


class Evaluation(object):

    def make_samples(self):
        raise NotImplementedError("Needs to implemented this method")


def distance(v1, v2, d_type='d1'):
    assert v1.shape == v2.shape, "shape of two vectors need to be same!"

    if d_type == 'd1':
        return np.sum(np.absolute(v1 - v2))
    elif d_type == 'd2':
        return np.sum((v1 - v2) ** 2)
    elif d_type == 'd2-norm':
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == 'd3':
        pass
    elif d_type == 'd4':
        pass
    elif d_type == 'd5':
        pass
    elif d_type == 'd6':
        pass
    elif d_type == 'd7':
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == 'd8':
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == 'cosine':
        return spatial.distance.cosine(v1, v2)
    elif d_type == 'square':
        return np.sum((v1 - v2) ** 2)


def AP(label, results, sort=True):
    ''' infer a query, return it's ap  计算对于特定的标签时，result的精度
    例如输入的result里有3个类别，其中一个类别与label相同，则精度即是1

      arguments
        label  : query's class
        results: a dict with two keys, see the example below
                 {
                   'dis': <distance between sample & query>,
                   'cls': <sample's class>
                 }
        sort   : sort the results by distance
    '''
    if sort:
        results = sorted(results, key=lambda x: x['dis'])
    precision = []
    hit = 0
    for i, result in enumerate(results):
        if result['cls'] == label:
            hit += 1
            precision.append(hit / (i + 1.))
    if hit == 0:
        return 0.
    return np.mean(precision)


def infer(query, samples=None, db=None, sample_db_fn=None, depth=None, return_img=False, d_type='d1'):
    ''' infer a query, return it's ap  推断输入值query与库中其他图片的距离，按升序排序并输出对每个组的精度计算。

      arguments
        query       : a dict with three keys, see the template
                      {
                        'img': <path_to_img>,
                        'cls': <img class>,
                        'hist' <img histogram>
                      }
        samples     : a list of {
                                  'img': <path_to_img>,
                                  'cls': <img class>,
                                  'hist' <img histogram>
                                }
        db          : an instance of class Database
        sample_db_fn: a function making samples, should be given if Database != None
        depth       : retrieved depth during inference, the default depth is equal to database size
        d_type      : distance type
        return_img  : 控制输出的results中包不包含库中图片的源地址
    '''
    assert samples != None or (
                db != None and sample_db_fn != None), "need to give either samples or db plus sample_db_fn"
    if db:
        samples = sample_db_fn(db)

    q_img, q_cls, q_hist = query['img'], query['cls'], query['hist']
    # 这里获取的是待计算值（单个的图片）的信息
    results = []
    for idx, sample in enumerate(samples):
        # 这里是对样本中的每个值计算待计算向量与这些值的距离
        s_img, s_cls, s_hist = sample['img'], sample['cls'], sample['hist']
        if q_img == s_img:
            continue
        results.append({
            'dis': distance(q_hist, s_hist, d_type=d_type),
            'cls': s_cls,
            **({'img': s_img} if return_img else {})
        })
    results = sorted(results, key=lambda x: x['dis'])
    # 对results内的字典，按距离升序排序
    if depth and depth <= len(results):
        results = results[:depth]
    ap = AP(q_cls, results, sort=False)

    return ap, results


def evaluate(db, sample_db_fn, depth=None, d_type='d1'):
    ''' infer the whole database

      arguments
        db          : an instance of class Database
        sample_db_fn: a function making samples, should be given if Database != None
        depth       : retrieved depth during inference, the default depth is equal to database size
        d_type      : distance type
    '''
    classes = db.get_class()
    ret = {c: [] for c in classes}

    samples = sample_db_fn(db)
    for query in samples:
        ap, _ = infer(query, samples=samples, depth=depth, d_type=d_type)
        ret[query['cls']].append(ap)

    return ret


def creat_feature(db, f_class=None, f_instance=None):
    '''生成某种方法所计算的图像特征值，该函数从evaluate_class中独立，这是因为我们需要只生成特征向量而不求其准度'''
    assert f_class or f_instance, "needs to give class_name or an instance of class"

    if f_class:
        f = f_class()
    elif f_instance:
        f = f_instance
    samples = f.make_samples(db)

    return samples


def evaluate_class(db, f_class=None, f_instance=None, depth=None, d_type='d1'):
    ''' infer the whole database  对整个数据库进行评估

        这个项目是调用这个函数生成特征向量数据的，然后这个函数又是调用各个方法中的makesample方法实现的，
        然后我想实现什么呢？
        我也不知道啊

      arguments
        db     : an instance of class Database
        f_class: a class that generate features, needs to implement make_samples method
        depth  : retrieved depth during inference, the default depth is equal to database size
        d_type : distance type
    '''

    classes = db.get_class()
    ret = {c: [] for c in classes}
    # # 获取所有图像种类的类名

    samples = creat_feature(db, f_class, f_instance)

    # 从缓存文件或当场计算库中所有图像的特征向量
    for query in tqdm(samples):
        ap, _ = infer(query, samples=samples, depth=depth, d_type=d_type)
        ret[query['cls']].append(ap)
        # 在ret中写入精度

    return ret
