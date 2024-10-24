# -*- coding: utf-8 -*-

import math
from pathlib import Path

import numpy as np
from PIL import Image
import pickle
from skimage import color
from skimage.feature import daisy
from tqdm import tqdm

from DB import Database
from evaluate import evaluate_class

n_slice = 2
n_orient = 8
step = 10
radius = 20  ## 本来是30，因为对于我的数据集30过大了，故改为20
rings = 2
histograms = 6
h_type = 'region'
d_type = 'd1'

depth = 3

R = (rings * histograms + 1) * n_orient

''' MMAP
     depth
      depthNone, daisy-region-n_slice2-n_orient8-step10-radius30-rings2-histograms6, distance=d1, MMAP 0.162806083971
      depth100,  daisy-region-n_slice2-n_orient8-step10-radius30-rings2-histograms6, distance=d1, MMAP 0.269333190731
      depth30,   daisy-region-n_slice2-n_orient8-step10-radius30-rings2-histograms6, distance=d1, MMAP 0.388199474789
      depth10,   daisy-region-n_slice2-n_orient8-step10-radius30-rings2-histograms6, distance=d1, MMAP 0.468182738095
      depth5,    daisy-region-n_slice2-n_orient8-step10-radius30-rings2-histograms6, distance=d1, MMAP 0.497688888889
      depth3,    daisy-region-n_slice2-n_orient8-step10-radius30-rings2-histograms6, distance=d1, MMAP 0.499833333333
      depth1,    daisy-region-n_slice2-n_orient8-step10-radius30-rings2-histograms6, distance=d1, MMAP 0.448

      (exps below use depth=None)

     d_type
      daisy-global-n_orient8-step180-radius58-rings2-histograms6, distance=d1, MMAP 0.101883969577
      daisy-global-n_orient8-step180-radius58-rings2-histograms6, distance=cosine, MMAP 0.104779921854

     h_type
      daisy-global-n_orient8-step10-radius30-rings2-histograms6, distance=d1, MMAP 0.157738278588
      daisy-region-n_slice2-n_orient8-step10-radius30-rings2-histograms6, distance=d1, MMAP 0.162806083971
'''

# cache dir
cache_dir = 'cache'
if not Path(cache_dir).exists():
    Path(cache_dir).mkdir()


class Daisy(object):

    def histogram(self, input, type=h_type, n_slice=n_slice, normalize=True):
        ''' count img histogram

          arguments
            input    : a path to a image or a numpy.ndarray
            type     : 'global' means count the histogram for whole image
                       'region' means count the histogram for regions in images, then concatanate all of them
            n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
            normalize: normalize output histogram

          return
            type == 'global'
              a numpy array with size R
            type == 'region'
              a numpy array with size n_slice * n_slice * R

            #R = (rings * histograms + 1) * n_orient#
        '''
        if isinstance(input, np.ndarray):  # examinate input type
            img = input.copy()
        else:
            img = Image.open(input, mode='r').convert('RGB')  # scipy.misc.imread 已被移除，替换为Image.open(img_path)
            img = np.array(img)  # 同步修改将图片转化为ndarray
        height, width, channel = img.shape

        P = math.ceil((height - radius * 2) / step)
        Q = math.ceil((width - radius * 2) / step)
        assert P > 0 and Q > 0, "input image size need to pass this check"

        if type == 'global':
            hist = self._daisy(img)

        elif type == 'region':
            hist = np.zeros((n_slice, n_slice, R))
            h_silce = np.around(np.linspace(0, height, n_slice + 1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(0, width, n_slice + 1, endpoint=True)).astype(int)

            for hs in range(len(h_silce) - 1):
                for ws in range(len(w_slice) - 1):
                    img_r = img[h_silce[hs]:h_silce[hs + 1], w_slice[ws]:w_slice[ws + 1]]  # slice img to regions
                    hist[hs][ws] = self._daisy(img_r)

        if normalize:
            hist /= np.sum(hist)

        return hist.flatten()

    def _daisy(self, img, normalize=True):
        image = color.rgb2gray(img)
        descs = daisy(image, step=step, radius=radius, rings=rings, histograms=histograms, orientations=n_orient)
        descs = descs.reshape(-1, R)  # shape=(N, R)
        hist = np.mean(descs, axis=0)  # shape=(R,)

        if normalize:
            hist = np.array(hist) / np.sum(hist)

        return hist

    def extract_features(self, db):
        samples = []
        data = db.get_data()
        for d in tqdm(data.itertuples(), desc='Extracting features'):
            d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
            d_hist = self.histogram(d_img, type=h_type, n_slice=n_slice)
            ## 对于非全局模式，请留意图片大小，图片太小可能会导致该模式出bug
            ## 这里的问题，我tm调试了2个下午，一步一步步进，找死我了

            samples.append({
                'img': d_img,
                'cls': d_cls,
                'hist': d_hist
            })
        return samples

    def make_samples(self, db, verbose=True):
        if h_type == 'global':
            sample_cache = "daisy-{}-n_orient{}-step{}-radius{}-rings{}-histograms{}".format(h_type, n_orient, step,
                                                                                             radius, rings, histograms)
        elif h_type == 'region':
            sample_cache = "daisy-{}-n_slice{}-n_orient{}-step{}-radius{}-rings{}-histograms{}".format(h_type, n_slice,
                                                                                                       n_orient, step,
                                                                                                       radius, rings,
                                                                                                       histograms)

        try:
            samples = pickle.load(open(Path(cache_dir)/ Path(sample_cache), "rb"))
            for sample in samples:
                sample['hist'] /= np.sum(sample['hist'])  # normalize
            if verbose:
                print("Using cache..., config=%s, distance=%s" % (sample_cache, d_type))
        except:
            if verbose:
                print("Counting histogram..., config=%s, distance=%s" % (sample_cache, d_type))

            samples = self.extract_features(db)

            pickle.dump(samples, open(Path(cache_dir)/ Path(sample_cache), "wb"))

        return samples


if __name__ == "__main__":
    db = Database()

    # evaluate database
    APs = evaluate_class(db, f_class=Daisy, d_type=d_type, depth=depth)
    cls_MAPs = []
    print("depth=%s"%depth)
    for cls, cls_APs in APs.items():
        MAP = np.mean(cls_APs)
        print("Class {}, MAP {}".format(cls, MAP))
        cls_MAPs.append(MAP)
    print("MMAP", np.mean(cls_MAPs))
