import numpy as np

class Checker:
    @staticmethod
    def compare(result, expected, posValue = 255):
        assert(len(result.shape) == 2 and len(expected.shape) == 2)
        assert(result.shape == expected.shape)

        height, width = result.shape
        confusionMatrix = dict.fromkeys(["tp", "tn", "fp", "fn"], 0)
        for x in range(height):
            for y in range(width):
                if (expected[x,y] == posValue):
                    if (result[x, y] == expected[x, y]):
                        confusionMatrix["tp"] += 1
                    else:
                        confusionMatrix["fn"] += 1
                else:
                    if (result[x, y] == expected[x, y]):
                        confusionMatrix["tn"] += 1
                    else:
                        confusionMatrix["fp"] += 1
        return confusionMatrix
        
    @staticmethod
    def createEmpty():
        return dict.fromkeys(["tp", "tn", "fp", "fn"], 0)

    @staticmethod
    def evaluate(confusionMatrix):
        tp = confusionMatrix["tp"]
        tn = confusionMatrix["tn"]
        fp = confusionMatrix["fp"]
        fn = confusionMatrix["fn"]

        acc = (tp + tn) / (tp + tn + fp + fn)
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)

        return (acc, sn, sp)

if __name__ == '__main__':
    import cv2
    testRes = './resources/CHASE/results1/Image_01L_1stHO.png'
    resImg = cv2.imread(testRes, cv2.IMREAD_GRAYSCALE)
    print(resImg.shape)
    fullNegative = np.zeros(resImg.shape)
    fullPositive = np.ones(resImg.shape) * 255
    testFN = Checker.compare(fullNegative, resImg)
    testFP = Checker.compare(fullPositive, resImg)
    testCorrect = Checker.compare(resImg, resImg)
    print(testFN)
    print(Checker.evaluate(testFN))
    print(testFP)
    print(Checker.evaluate(testFP))
    print(testCorrect)
    print(Checker.evaluate(testCorrect))