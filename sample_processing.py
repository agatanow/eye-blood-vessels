import cv2
import numpy as np
from matplotlib import pyplot as plt

class Sample:
    @staticmethod
    def calcHuMoments(roi):
        gray = roi #cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        return cv2.HuMoments(cv2.moments(gray)).flatten()

    # moments sequence: mu20, mu11, mu02, mu30, mu21, mu12, mu03
    @staticmethod
    def calcCentralMoments(roi):
        gray = roi #cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        moments = cv2.moments(gray)
        centralMoments = np.array([
            moments['mu20'],
            moments['mu11'],
            moments['mu02'],
            moments['mu30'],
            moments['mu21'],
            moments['mu12'],
            moments['mu03']
        ])
        return centralMoments

    @staticmethod
    def calcVariance(roi):
        return np.square(cv2.meanStdDev(roi)[1].flatten())

    @staticmethod
    def calcMean(roi):
        return cv2.meanStdDev(roi)[0].flatten()

    @staticmethod
    def calcAllFeatures(roi, flatten=False):
        huMoments = Sample.calcHuMoments(roi)
        centralMoments = Sample.calcCentralMoments(roi)
        variance = Sample.calcVariance(roi)
        mean = Sample.calcMean(roi)
        return np.array([*variance, *mean, *huMoments, *centralMoments]) if flatten else [variance, mean, huMoments, centralMoments]

    @staticmethod
    def cut_roi(roi):
        return roi[1:-1,1:-1]

    @staticmethod
    def calcAllFeatures2(roi):
        result=[]
        while(roi.shape[0]>1):
            result = [*result, *Sample.calcAllFeatures(roi, flatten=True)]
            roi = Sample.cut_roi(roi)
        return np.array(result)



if __name__ == '__main__':
    testImg = './resources/CHASE/original/Image_01L.jpg'
    testRes = './resources/CHASE/results1/Image_01L_1stHO.png'

    img = cv2.imread(testImg)[:,:,::-1]
    resImg = cv2.imread(testRes, cv2.IMREAD_GRAYSCALE)
    roi = img[330:339, 620:629]
    resROI = resImg[330:339, 620:629]
    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(5, 5))
    print(Sample.calcVariance(roi))
    print(Sample.calcMean(roi))
    print(Sample.calcHuMoments(roi))
    print(Sample.calcCentralMoments(roi))
    print(Sample.calcAllFeatures(roi))
    print(Sample.calcAllFeatures(roi, True))
    ax[0].imshow(roi, aspect='auto')
    ax[1].imshow(resROI, aspect='auto', cmap='gray')
    plt.show()
