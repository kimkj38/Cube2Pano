import numpy as np
from im2Sphere import im2Sphere

def combineViews(Imgs, width, height, depth=False):
    if not depth:
        depth = False

    panoout = np.zeros((height, width, Imgs[0]['img'].shape[2])) #(1024, 2048, 3)
    panowei = np.zeros((height, width, Imgs[0]['img'].shape[2]))
    imgNum = len(Imgs)

    for i in range(imgNum):
        sphereImg, validMap = im2Sphere(Imgs[i]['img'], Imgs[i]['fov'], width, height, Imgs[i]['vx'], Imgs[i]['vy'], 'linear')
        # print(np.sum(validMap))
        sphereImg[~validMap] = 0
        panoout = panoout + sphereImg
        panowei = panowei + validMap

    panoout[panowei == 0] = 0
    panowei[panowei == 0] = 1
    # panoout = panoout / np.double(panowei)

    if depth:
        panodepth = np.zeros((height, width))
        panowei = np.zeros((height, width))

        for i in range(imgNum):
            sphereImg, validMap = im2Sphere(Imgs[i]['dep'], Imgs[i]['fov'], width, height, Imgs[i]['vx'], Imgs[i]['vy'], 'nearest')
            sphereImg[~validMap] = 0
            panodepth = panodepth + sphereImg
            panowei = panowei + validMap

        panodepth[panowei == 0] = 0
        panowei[panowei == 0] = 1
        panodepth = panodepth / np.double(panowei)
    else:
        panodepth = 0

    return panoout, panodepth
