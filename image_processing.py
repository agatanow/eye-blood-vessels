#!/usr/bin/env python
import cv2
from checker import Checker
import numpy as np
from skimage.filters import frangi, threshold_triangle
from skimage import util 
from skimage import img_as_ubyte
from matplotlib import pyplot as plt

class ImagePreprocessor:
    __kernel = (9, 9)
    __clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(8,8))
    
    @staticmethod
    def loadImage(path, asGray = False):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE) if asGray else cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def toGray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def preprocess(img):
        imgG = img[:,:,1]
        norm = ImagePreprocessor.__clahe.apply(imgG)
        return norm

class ImageProcessor:
    @staticmethod
    def meanColor(hist, start, end):
        col = 0.
        count = 0.
        for x in range(start, end):
            col += x * hist[x]
            count += hist[x]
        return int(col // count)

    @staticmethod
    def process(img):
        height, width = img.shape
        mean = cv2.blur(img, (9, 9))
        diff = np.zeros(img.shape)
        for x in range(height):
            for y in range(width):
                diff[x, y] = mean[x, y] - img[x, y] if mean[x, y] > img[x, y] else 0   
        
        result = np.zeros(img.shape)
        for x in range(height):
            for y in range(width):
                result[x, y] = diff[x, y] if diff[x, y] > 0 else 0
        result = result.astype('uint8')
        hist = cv2.calcHist([result], [0], None, [256], [0, 256]).flatten()
        c_min = np.min(result)
        c_max = np.max(result)

        t0 = (c_min + c_max) // 2
        tk = (ImageProcessor.meanColor(hist, 0, t0) + ImageProcessor.meanColor(hist, t0, len(hist))) // 2
        while abs(t0 - tk) > 0:
            t0 = tk
            tk = (ImageProcessor.meanColor(hist, 0, t0) + ImageProcessor.meanColor(hist, t0, len(hist))) // 2

        for x in range(height):
            for y in range(width):
                result[x, y] = 255 if result[x,y] > tk else 0

        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, (9,9), iterations = 1)

        return result

    @staticmethod
    def __labDistance(first, second):
        return np.sqrt(np.sum((first - second)**2))

    @staticmethod
    def neighbourPixels(img, refColor, treshold = 40, kernel=(9,9)):
        height, width, _ = img.shape
        imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        refColorLab = cv2.cvtColor(refColor, cv2.COLOR_RGB2LAB)
        mask = np.zeros((height, width))
        for x in range(height):
            for y in range(width):
                if (ImageProcessor.__labDistance(imgLab[x,y], refColorLab[0,0]) < 4):
                    different = 0
                    for xx in range(kernel[0]//2):
                        for yy in range(kernel[1]//2):
                            if (xx == 0 and yy == 0):
                                continue
                            if (x - xx >= 0 and y - yy >= 0):
                                if (ImageProcessor.__labDistance(imgLab[x,y], imgLab[x - xx, y - yy]) > 0):
                                    different += 1
                            if (x + xx < height and y + yy < width):
                                if (ImageProcessor.__labDistance(imgLab[x,y], imgLab[x + xx, y + yy]) > 0):
                                    different += 1
                    if (different > treshold):
                        mask[x,y] = 255
        return mask

    @staticmethod
    def neighbourPixelsRGB(img, colTresh, treshold = 40, kernel=(9,9)):
        height, width, _ = img.shape
        mask = np.zeros((height, width))
        for x in range(height):
            for y in range(width):
                if img[x,y,0] > np.max(img[x,y, 1:]) and (img[x,y,0] - np.max(img[x,y, 1:]) > colTresh):
                    different = 0
                    for xx in range(kernel[0]//2):
                        for yy in range(kernel[1]//2):
                            if (xx == 0 and yy == 0):
                                continue
                            if (x - xx >= 0 and y - yy >= 0):
                                if (ImageProcessor.__labDistance(img[x,y], img[x - xx, y - yy]) > 10):
                                    different += 1
                            if (x + xx < height and y + yy < width):
                                if (ImageProcessor.__labDistance(img[x,y], img[x + xx, y + yy]) > 10):
                                    different += 1
                    if (different > treshold):
                        mask[x,y] = 255
        return mask               

    @staticmethod
    def morphFilter(img):
        structElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, structElement)
        open = cv2.morphologyEx(close, cv2.MORPH_OPEN, structElement)
        return img - close
        
    @staticmethod
    def kirchFilter(img, threshold):
        g1 = np.array([5, 5, 5, 
                      -3, 0, -3, 
                      -3, -3, -3]).reshape((3,3))

        g2 = np.array([5, 5, -3, 
                       5, 0, -3, 
                       -3, -3, -3]).reshape((3,3))

        g3 = np.array([5, -3, -3, 
                       5, 0, -3, 
                       5, -3, -3]).reshape((3,3))

        g4 = np.array([-3, -3, -3, 
                       5, 0, -3, 
                       5, 5, -3]).reshape((3,3))

        g5 = np.array([-3, -3, -3, 
                       -3, 0, -3, 
                       5, 5, 5]).reshape((3,3))

        g6 = np.array([-3, -3, -3, 
                       -3, 0, 5, 
                       -3, 5, 5]).reshape((3,3))

        g7 = np.array([-3, -3, 5, 
                       -3, 0, 5, 
                       -3, -3, 5]).reshape((3,3))

        g8 = np.array([-3, 5, 5, 
                       -3, 0, 5, 
                       -3, -3, -3]).reshape((3,3))
        
        f1 = cv2.filter2D(img, -1, g1)
        f2 = cv2.filter2D(img, -1, g2)
        f3 = cv2.filter2D(img, -1, g3)
        f4 = cv2.filter2D(img, -1, g4)
        f5 = cv2.filter2D(img, -1, g5)
        f6 = cv2.filter2D(img, -1, g6)
        f7 = cv2.filter2D(img, -1, g7)
        f8 = cv2.filter2D(img, -1, g8)

        return np.maximum.reduce([f1, f2, f3, f4, f5, f6, f7, f8]) > threshold

    @staticmethod
    def adaptiveFilter(img):
        filtered = cv2.bilateralFilter(img, 5, 5, 5)
        #return filtered
        treshold = cv2.bitwise_not(cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2))
        res = cv2.dilate(treshold, np.ones((3, 3), np.uint8), iterations=1)
        res = cv2.erode(res, np.ones((3, 3), np.uint8), iterations=1)
        # blurred = cv2.GaussianBlur(img, (5, 5), 0)
        # res = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)
        return res

class ImagePostprocessor:

    @staticmethod
    def calibrateMask(img):
        r, g, b = cv2.split(img)
        c_sum = r.astype('uint16') + g + b
        mask = (np.bitwise_not(c_sum > 70) * 255).astype('uint8') 
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations = 3)
        mask = np.bitwise_not(mask)
        return mask

    @staticmethod
    def process(img, mask = None):
        res = img.copy()
        if (mask is not None and mask.shape == img.shape):
            res = cv2.bitwise_and(res, res, mask = mask)
        return res

if __name__ == '__main__':
    from db_controller import DbController
    from checker import Checker
    db = DbController()
    results = "./results/image_processing/"
    dataset = db.get_dataset(DbController.STARE)
    statistics = []
    for links in dataset:
        imgPath = links[0]
        name = imgPath[imgPath.rfind('/') + 1:imgPath.rfind('.')]
        print(name)

        resPath = links[1]
        img = ImagePreprocessor.loadImage(imgPath)
        exp = ImagePreprocessor.loadImage(resPath, asGray=True)

        prep = ImagePreprocessor.preprocess(img)

        out = ImageProcessor.process(prep)

        mask = ImagePostprocessor.calibrateMask(img)
        post = ImagePostprocessor.process(out, mask=mask)

        color = np.zeros(img.shape).astype('uint8')
        color[:,:,2] = 255
        color = cv2.bitwise_and(color, color, mask = post)
        res = cv2.bitwise_and(img, img, mask = cv2.bitwise_not(post))
        res = cv2.add(res, color)

        fig, ax = plt.subplots(4, 2, sharex=True, figsize=(9, 9))
        ax[0,0].set_title('Img')
        ax[0,1].set_title('Green channel')
        ax[1,0].set_title('Preprocess')
        ax[1,1].set_title('Process')
        ax[2,0].set_title('Mask')
        ax[2,1].set_title('Postprocess')
        ax[3,0].set_title('Expected')
        ax[3,1].set_title('Result')

        ax[0,0].imshow(img, aspect='auto')
        ax[0,1].imshow(img[:,:,1], aspect='auto', cmap='gray')
        ax[1,0].imshow(prep, aspect='auto', cmap='gray')
        ax[1,1].imshow(out, aspect='auto', cmap='gray')
        ax[2,0].imshow(mask, aspect='auto', cmap='gray')
        ax[2,1].imshow(post, aspect='auto', cmap='gray')
        ax[3,0].imshow(exp, aspect='auto', cmap='gray')
        ax[3,1].imshow(res, aspect='auto')

        cv2.imwrite(results + "mask/" + name + ".png", post)
        cv2.imwrite(results + "img/" + name + ".png", res[:,:,::-1])
        plt.savefig(results + "detail/" + name + ".pdf")

        confusionMatrix = Checker.compare(post, exp)
        statistics.append([name, confusionMatrix])
    with open(results + "results.txt", "w") as res_file:
        avg_stat = Checker.createEmpty()
        for statistic in statistics:
            name = statistic[0]
            cmatrix = statistic[1]
            avg_stat["tp"] += cmatrix["tp"]
            avg_stat["fp"] += cmatrix["fp"]
            avg_stat["tn"] += cmatrix["tn"]
            avg_stat["fn"] += cmatrix["fn"]
            stats = Checker.evaluate(cmatrix)
            res_file.write("File: " + name + "\n")
            res_file.write("- Confusion Matrix\n")
            res_file.write("-- True positive: " + str(cmatrix["tp"]) + "\n")
            res_file.write("-- False positive: " + str(cmatrix["fp"]) + "\n")
            res_file.write("-- True negative: " + str(cmatrix["tn"]) + "\n")
            res_file.write("-- False negative: " + str(cmatrix["fn"]) + "\n")
            res_file.write("- Measure\n")
            res_file.write("-- Accuracy: " + str(stats[0]) + "\n")
            res_file.write("-- Sensitivity: " + str(stats[1]) + "\n")
            res_file.write("-- Specificity: " + str(stats[2]) + "\n")
        stats = Checker.evaluate(avg_stat)
        res_file.write("Total\n")
        res_file.write("- Confusion Matrix\n")
        res_file.write("-- True positive: " + str(avg_stat["tp"]) + "\n")
        res_file.write("-- False positive: " + str(avg_stat["fp"]) + "\n")
        res_file.write("-- True negative: " + str(avg_stat["tn"]) + "\n")
        res_file.write("-- False negative: " + str(avg_stat["fn"]) + "\n")
        res_file.write("- Measure\n")
        res_file.write("-- Accuracy: " + str(stats[0]) + "\n")
        res_file.write("-- Sensitivity: " + str(stats[1]) + "\n")
        res_file.write("-- Specificity: " + str(stats[2]) + "\n")
