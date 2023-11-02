import numpy as np
from scipy.interpolate import interp2d
import cv2

def warpImageFast(im, XXdense, YYdense, method):

    minX = max(1, int(np.floor(np.min(XXdense))) - 1)
    minY = max(1, int(np.floor(np.min(YYdense))) - 1)

    maxX = min(im.shape[1], int(np.ceil(np.max(XXdense))) + 1)
    maxY = min(im.shape[0], int(np.ceil(np.max(YYdense))) + 1)

    im = im[minY-1:maxY, minX-1:maxX, :]

    im_warp = np.zeros((XXdense.shape[0],XXdense.shape[1], im.shape[2]))
    # h = XXdense.shape[0]
    # w = XXdense.shape[1]

    # XXdense = XXdense.flatten()
    # YYdense = YYdense.flatten()
    # partition = int(XXdense.shape[0]/8)


    for c in range(im.shape[2]):
        im_warp[:, :, c] = cv2.remap(im[:, :, c], (XXdense - minX).astype(np.float32), (YYdense - minY).astype(np.float32), cv2.INTER_LINEAR)
    

    # for c in range(im.shape[2]):
    #     f = interp2d(np.arange(1, im.shape[1] + 1), np.arange(1, im.shape[0] + 1), im[:, :, c], kind=method)
    #     # k = f(XXdense[:partition], YYdense[:partition])
    #     # im_warp[:, c] = f(XXdense - minX + 1, YYdense - minY + 1)
    #     x_coords = (XXdense - minX + 1).ravel()  # 1D로 변환
    #     y_coords = (YYdense - minY + 1).ravel()  # 1D로 변환
    #     interpolated = f(x_coords, y_coords)
    #     im_warp[:,:,c] = interpolated.reshape(XXdense.shape)  # 다시 2D 형태로 변환
        
    # im_warp = im_warp.reshape((h,w,im.shape[2]))

    return im_warp
