# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import distance, evaluate_class
from DB import Database

from six.moves import cPickle
import numpy as np
from PIL import Image
import itertools
import os

from tqdm import tqdm


# configs for histogram
n_bin   = 12        # histogram bins
n_slice = 3         # slice image
h_type  = 'region'  # global or region
d_type  = 'd1'      # distance type

depth   = 3         # retrieved depth, set to None will count the ap for whole database
# depth 指的是最终计算MP时保留的距离最小的几个实体，值为几即保留几个

''' MMAP
     depth
      depthNone, region,bin12,slice3, distance=d1, MMAP 0.273745840034
      depth100,  region,bin12,slice3, distance=d1, MMAP 0.406007856783
      depth30,   region,bin12,slice3, distance=d1, MMAP 0.516738512679
      depth10,   region,bin12,slice3, distance=d1, MMAP 0.614047666604
      depth5,    region,bin12,slice3, distance=d1, MMAP 0.650125
      depth3,    region,bin12,slice3, distance=d1, MMAP 0.657166666667
      depth1,    region,bin12,slice3, distance=d1, MMAP 0.62

     (exps below use depth=None)
     
     d_type
      global,bin6,d1,MMAP 0.242345913685
      global,bin6,cosine,MMAP 0.184176505586

     n_bin
      region,bin10,slice4,d1,MMAP 0.269872790396
      region,bin12,slice4,d1,MMAP 0.271520862017

      region,bin6,slcie3,d1,MMAP 0.262819311357
      region,bin12,slice3,d1,MMAP 0.273745840034

     n_slice
      region,bin12,slice2,d1,MMAP 0.266076627332
      region,bin12,slice3,d1,MMAP 0.273745840034
      region,bin12,slice4,d1,MMAP 0.271520862017
      region,bin14,slice3,d1,MMAP 0.272386552594
      region,bin14,slice5,d1,MMAP 0.266877181379
      region,bin16,slice3,d1,MMAP 0.273716788003
      region,bin16,slice4,d1,MMAP 0.272221031804
      region,bin16,slice8,d1,MMAP 0.253823360098

     h_type
      region,bin4,slice2,d1,MMAP 0.23358615622
      global,bin4,d1,MMAP 0.229125435746
      
      MMAP 0.3656666666666667
'''

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)


class Color(object):

  def histogram(self, input, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
    ''' count img color histogram 统计图片中的颜色直方
  
      arguments
        input    : a path to a image or a numpy.ndarray
        n_bin    : number of bins for each channel
        type     : 'global' means count the histogram for whole image
                   'region' means count the histogram for regions in images, then concatanate all of them
        n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
        normalize: normalize output histogram
  
      return
        type == 'global'
          a numpy array with size n_bin ** channel
        type == 'region'
          a numpy array with size n_slice * n_slice * (n_bin ** channel)
    '''
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = Image.open(input, mode='r').convert('RGB')  # scipy.misc.imread 已被移除，替换为Image.open(img_path)
      img = np.array(img)   # 同步修改将图片转化为ndarray
    height, width, channel = img.shape
    bins = np.linspace(0, 256, n_bin+1, endpoint=True)  # slice bins equally for each channel
    # bins 是一个0-256包含端点的均匀分割为13份的数组。[0,21.3,42.6,64,85,106,128,149,170,192,213,234,256]
  
    if type == 'global':
      hist = self._count_hist(img, n_bin, bins, channel)
      # print(hist)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, n_bin ** channel))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._count_hist(img_r, n_bin, bins, channel)
  
    if normalize:
      hist /= np.sum(hist)
    # 这个正则化居然是和为1，不应该是每个值在0-1之间嘛
    # print(hist)
  
    return hist.flatten()
  
  
  def _count_hist(self, input, n_bin, bins, channel):
    '''对图像中的颜色直方进行统计'''
    img = input.copy()
    bins_idx = {key: idx for idx, key in enumerate(itertools.product(np.arange(n_bin), repeat=channel))}  # permutation of bins
    # itertools.product(np.arange(n_bin), repeat=channel)) 是一个生成器，例如 itertools.product(np.arange(2), repeat=1))用for
    # 读出来的是简单的 0，1。如果是 itertools.product(np.arange(2), repeat=2))读出来的是00，01，10，11，上面的生成器生成出来的是000，
    # 001一直至111111，然后利用enumerate将其与index打包输出被集合bins_idx接收

    hist = np.zeros(n_bin ** channel)
  
    # cluster every pixels
    for idx in range(len(bins)-1):
      img[(input >= bins[idx]) & (input < bins[idx+1])] = idx
    # 这一步是将图像中的rgb值从rgb编码为直方图中的值，例如11为234-265，0为0-21，一张图为[[[1,1,250],[250,250,1]],[[250,250,1]],[[1,1,250]]]，
    # 则该图转化为[[[0,0,11],[11,11,0]],[[11,11,0]],[[0,0,11]]]

    # add pixels into bins
    height, width, _ = img.shape
    for h in range(height):
      for w in range(width):
        b_idx = bins_idx[tuple(img[h,w])]
        hist[b_idx] += 1
    # 这里是在循环求这张图中的每种rgb的数量，例如一张图为[[[0,0,1],[0,0,2]],[[0,0,1]],[[0,0,3]]]，然后再定义一个列表hist最初是[0,0,0]
    # 这个列表第一个值代表[0,0,0]出现的次数，第二个值代表[0,0,1]出现的次数，依此类推，最后输出的hist是[0,2,1,1]

    return hist
  
  
  def make_samples(self, db, verbose=True):
    '''用于制作样本？跟废话似的
    '''
    if h_type == 'global':
      sample_cache = "histogram_cache-{}-n_bin{}".format(h_type, n_bin)
    elif h_type == 'region':
      sample_cache = "histogram_cache-{}-n_bin{}-n_slice{}".format(h_type, n_bin, n_slice)
    
    try:
      samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb"))
      if verbose:
        print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
    except:
      if verbose:
        print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
      samples = []
      data = db.get_data()
      for d in tqdm(data.itertuples()):
        d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
        d_hist = self.histogram(d_img, type=h_type, n_bin=n_bin, n_slice=n_slice)
        samples.append({
                        'img':  d_img, 
                        'cls':  d_cls, 
                        'hist': d_hist
                      })
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb"))
  
    return samples


if __name__ == "__main__":
  db = Database()
  data = db.get_data()
  color = Color()

  # test normalize
  hist = color.histogram(data.iloc[0,0], type='global')
  assert hist.sum() - 1 < 1e-9, "normalize false"

  # test histogram bins
  def sigmoid(z):
    a = 1.0 / (1.0 + np.exp(-1. * z))
    return a
  np.random.seed(0)
  IMG = sigmoid(np.random.randn(2,2,3)) * 255
  IMG = IMG.astype(int)
  hist = color.histogram(IMG, type='global', n_bin=4)
  assert np.equal(np.where(hist > 0)[0], np.array([37, 43, 58, 61])).all(), "global histogram implement failed"
  # equal中的内容，是说比较两个数组是否相同，例如[1,2]与[1,3]相比，输出是[True,False]，然后all是比较表内是否相同，这个例子中输出是False
  hist = color.histogram(IMG, type='region', n_bin=4, n_slice=2)
  assert np.equal(np.where(hist > 0)[0], np.array([58, 125, 165, 235])).all(), "region histogram implement failed"

  # examinate distance
  np.random.seed(1)
  IMG = sigmoid(np.random.randn(4,4,3)) * 255
  IMG = IMG.astype(int)
  hist = color.histogram(IMG, type='region', n_bin=4, n_slice=2)
  IMG2 = sigmoid(np.random.randn(4,4,3)) * 255
  IMG2 = IMG2.astype(int)
  hist2 = color.histogram(IMG2, type='region', n_bin=4, n_slice=2)
  assert distance(hist, hist2, d_type='d1') == 2, "d1 implement failed"
  assert distance(hist, hist2, d_type='d2-norm') == 2, "d2 implement failed"

  # evaluate database
  APs = evaluate_class(db, f_class=Color, d_type=d_type, depth=depth)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    print("Class {}, MAP {}".format(cls, MAP))
    cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))
  # 最后计算出每个类的ap已经总平均ap
