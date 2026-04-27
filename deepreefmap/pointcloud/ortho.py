import numpy as np
from sklearn.decomposition import PCA


def pca_ortho_projection(xyz: np.ndarray, rgb: np.ndarray, cls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2)
    xy = pca.fit_transform(xyz)
    xy -= xy.min(axis=0, keepdims=True)
    if xy.shape[0] == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)
    scale = max(1.0, np.percentile(xy, 99) / 2000.0)
    ij = np.floor(xy / scale).astype(np.int32)
    h = int(ij[:, 1].max() + 1)
    w = int(ij[:, 0].max() + 1)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    seg = np.zeros((h, w), dtype=np.uint8)
    img[ij[:, 1], ij[:, 0]] = rgb
    seg[ij[:, 1], ij[:, 0]] = cls
    return img, seg
