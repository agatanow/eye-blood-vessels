#!/usr/bin/env python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from db_controller import DbController
import cv2
from matplotlib import pyplot as plt
from random import randint,sample
import numpy as np
from sample_processing import Sample
from sklearn import svm
from image_processing import ImagePreprocessor, ImagePostprocessor
from checker import Checker
from sklearn.model_selection import cross_val_score
import pickle

class Neuron:
    def __init__(self):
        self.sample_size = 15
        self.sample_no = 10000
        self.data = [] #each row is a vector of features for one sample
        self.target = [] #each value is decision for a sample
        #random forest classificator
        self.clf = RandomForestClassifier()
        '''RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                            max_depth=2, max_features='auto', max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                            oob_score=False, random_state=0, verbose=0, warm_start=False)'''

    def image_sample_coords(self,image):
        tr=[]
        fs=[]
        for y in range(image.shape[0]-1-self.sample_size):
            for x in range(image.shape[1]-1-self.sample_size):
                if image[y][x]==255:
                    tr.append([y,x])
                else:
                    fs.append([y,x])
        con = np.concatenate((sample(tr,self.sample_no//2),sample(fs,self.sample_no//2)), axis=0)
        np.random.shuffle(con)
        return con

    def get_sample(self,img,y,x):
        return img[y:y + self.sample_size,x:x+self.sample_size]

    def sample_features(self, sample):
        return Sample.calcAllFeatures2(sample)

    def add_samples(self, org_path, res_path):
        img = ImagePreprocessor.loadImage(org_path)
        img = ImagePreprocessor.preprocess(img)
        res = ImagePreprocessor.loadImage(res_path, asGray=True)
        #coords = self.image_sample_coords(img.shape[0], img.shape[1])
        for c in self.image_sample_coords(res):
            #append new row of sample features
            sample = self.get_sample(img,c[0],c[1])
            self.data.append(self.sample_features(sample))
            #append decision for corresponding sample
            self.target.append(res[c[0] + self.sample_size//2, c[1] + self.sample_size//2]//255)

    def set_dataset(self, paths):
        self.data=[]
        self.target=[]
        for p in paths:
            self.add_samples(p[0],p[1])

    def train(self):
        self.clf.fit(self.data, self.target) #classification learn

    def test(self, org_path):
        img = ImagePreprocessor.loadImage(org_path)
        img = ImagePreprocessor.preprocess(img)
        result = []
        for y in range(0,img.shape[0]-self.sample_size-1):
            row = []
            for x in range(0,img.shape[1]-self.sample_size-1):
                decision = self.clf.predict([self.sample_features(self.get_sample(img,y,x))])
                row.append(decision[0])
            for x in range(img.shape[1]-self.sample_size-1,img.shape[1]):
                row.append(0)
            result.append(row)
            print(y)

        for y in range(img.shape[0]-self.sample_size-1,img.shape[0]):
            result.append([0 for x in range(0,img.shape[1])])
        return np.array(result)*255


    def k_fold_cv(self, k=5):
        return cross_val_score(self.clf, self.data, self.target, cv=k)

    def save_model(self):
        pickle.dump(self.clf, open('model.sav', 'wb'))

    def load_model(self):
        pickle.load(open('model.sav', 'rb'))

if __name__ == '__main__':
    base = DbController()
    ds = Neuron()
    ds.set_dataset(base.get_dataset(DbController.STARE)[5:])
    dataset = base.get_dataset(DbController.STARE)[:5]
    results = "./results/random_forrest/"
    k = 5
    exo = ds.k_fold_cv(10, k)
    ds.train()

#zapisac wyniki k cros walidacji i srednia do pliku
    with open(results + "results.txt", "w") as res_file:
        res_file.write("K-fold cross validation\n")
        res_file.write("- k:" + str(k) + "\n")
        res_file.write("- result:" + str(exo) + "\n")
        res_file.write("- mean:" + str(np.mean(exo)) + "\n")

#zapisac obrazki dla obrazow STARE[:5]
    statistics = []
    for links in dataset:
        imgPath = links[0]
        name = imgPath[imgPath.rfind('/') + 1:imgPath.rfind('.')]
        print(name)

        resPath = links[1]
        img = ImagePreprocessor.loadImage(imgPath)
        exp = ImagePreprocessor.loadImage(resPath, asGray=True)

        prep = ImagePreprocessor.preprocess(img)

        out = ds.test(imgPath).astype('uint8')

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
    with open(results + "results.txt", "a") as res_file:
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
