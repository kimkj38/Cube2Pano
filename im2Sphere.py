import numpy as np
from scipy.ndimage import map_coordinates
from warpImageFast import warpImageFast
# from scipy.interpolate import RegularGridInterpolator

def im2Sphere(im, imHoriFOV, sphereW, sphereH, x, y, method):
    # Map pixel in panorama to viewing direction
    TX, TY = np.meshgrid(np.arange(1,sphereW+1), np.arange(1,sphereH+1)) #(1024, 2048)
    TX = TX.flatten()
    TY = TY.flatten()

    # 각 픽셀을 spherical coordinate(radian)으로 바꿔줌
    ANGx = (TX - sphereW / 2 - 0.5) / sphereW * np.pi * 2
    ANGy = -(TY - sphereH / 2 - 0.5) / sphereH * np.pi

    # Compute the radius of the sphere
    imH, imW, _ = im.shape
    R = (imW / 2) / np.tan(imHoriFOV / 2)

    # Compute the coordinates of the sphere's center
    x0 = R * np.cos(y) * np.sin(x)
    y0 = R * np.cos(y) * np.cos(x)
    z0 = R * np.sin(y)

    # Define the plane function and view line
    alpha = np.cos(ANGy) * np.sin(ANGx)
    belta = np.cos(ANGy) * np.cos(ANGx)
    gamma = np.sin(ANGy)

    # solve for intersection of plane and viewing line: [x1 y1 z1]  
    division = x0 * alpha + y0 * belta + z0 * gamma
    
    
    x1 = R * R * alpha / division
    y1 = R * R * belta / division
    z1 = R * R * gamma / division

    # Compute the vector in the plane and positive x, y vectors
    vec = np.array([x1 - x0, y1 - y0, z1 - z0]) # (3, 2097152)
    vecposX = np.expand_dims(np.array([np.cos(x), -np.sin(x), 0]), 0) #(1,3)이 되어야 함
    deltaX = (vecposX @ vec) / np.sqrt(vecposX @ vecposX.T) #(1,2097152), (1,1) -> (1, 2097152)

    vecposY = np.cross(np.array([x0, y0, z0]), vecposX) # (1,3)
    deltaY = (vecposY @ vec) / np.sqrt(vecposY @ vecposY.T) # (1,2097152)

    # print(deltaY.reshape((1024,2048))[:3,:3]+ (imW+1)/2)

    # Convert to image coordinates
    Px = (deltaX.reshape((sphereH, sphereW)) + (imW + 1) / 2) # 1024, 2048
    Py = (deltaY.reshape((sphereH, sphereW)) + (imH + 1) / 2)
    
    # Warp image
    # Px = np.clip(Px, 0, imW - 1)
    # Py = np.clip(Py, 0, imH - 1)
    # warp_coords = (Py, Px)
    # sphereImg = map_coordinates(im, warp_coords, order=1, mode=method, cval=np.nan)
    sphereImg = warpImageFast(im, Px, Py, method)
    validMap = ~np.isnan(sphereImg[:, :, 0])

    if method == 'nearest':
        validMap[sphereImg[:, :, 0] == 0] = False

    

    division = division.reshape((validMap.shape[0], validMap.shape[1]))

    validMap[division < 0] = False
    # print(np.sum(validMap))
    validMap = np.repeat(validMap[:, :, np.newaxis], sphereImg.shape[2], axis=2)

    return sphereImg, validMap
