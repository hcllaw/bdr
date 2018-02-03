from functools import partial
import os
import warnings

import numpy as np
import pandas as pd

from bdr import Features


path = partial(os.path.join, os.path.dirname(__file__))


def load_faces(mmap_mode=None, drop_allzeros=True, load_their_probs=False):
    fn = path('grouped-4', 'data.csv.gz')
    meta = pd.read_csv(fn, parse_dates=['dob', 'photo_taken'])
    N = meta.shape[0]

    if load_their_probs:
        probs = np.load(path('all-feats/probs.npy'))
        with np.load(path('all-feats/meta.npz')) as m:
            order = {p: i for i, p in enumerate(m['full_path'])}
        meta['probs'] = meta.full_path.map(lambda x: probs[order[x]])

    name_dobs = meta[['name', 'dob']].drop_duplicates()
    inds = name_dobs.index.values
    n_pts = np.r_[np.diff(inds), N - inds[-1]]
    n_bags = n_pts.size

    onlies = [('name', object),
              ('dob', 'datetime64[D]'),
              ('gender', np.int8)]
    groups = ['pred', 'age', 'photo_taken', 'full_path',
              'face_score', 'second_face_score']
    if load_their_probs:
        groups.append('probs')

    metas = {k: np.empty(n_bags, dtype=object) for k in groups}
    for k, dt in onlies:
        metas[k] = np.empty(n_bags, dtype=dt)
    metas['mean_pred'] = np.empty(n_bags, dtype=np.float64)
    metas['y'] = np.empty(n_bags, dtype=np.float64)

    for i, (start, end) in enumerate(zip(inds, np.r_[inds[1:], N])):
        for k, dt in onlies:
            v = meta[k].values[start]
            if k == 'gender' and np.isnan(v):
                v = -1
            metas[k][i] = v
        for k in groups:
            v = meta[k].values[start:end]
            if k == 'photo_taken':
                v = v.astype('datetime64[D]')
            metas[k][i] = v
        metas['mean_pred'][i] = metas['pred'][i].mean()
        metas['y'][i] = metas['age'][i].mean()

    if mmap_mode is not None and drop_allzeros:
        warnings.warn("drop_allzeros makes mmap_mode pointless, ignoring")
        mmap_mode = None
    fn = path('grouped-4', 'feats.npy')
    t = '/tmp/feats.npy'
    if os.path.exists(t) and os.path.getsize(fn) == os.path.getsize(t):
        fn = t
    data = np.load(fn, mmap_mode=mmap_mode)
    if drop_allzeros:
        w = data.any(axis=0)
        data = data[:, w]
    return Features(data, n_pts, **metas)


def get_emp_cov():
    emp_cov = np.load(path('grouped-4/emp_cov.npy'))
    with np.load(path('grouped-4/emp_cov_eigs.npz')) as d:
        return emp_cov, d['eigvals'], d['eigvecs']
