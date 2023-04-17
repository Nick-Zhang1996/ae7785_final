# identify signs in a traditional CV way
import cv2
import sys
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pickle

class Signs:

    def __init__(self):
        self.image_dir = './2022Fimgs/'
        self.suffix = '.png'
        self.label2text = ['empty','left','right','do not enter','stop','goal']
        self.clf = None

    def saveModel(self):
        with open('svm.p','wb') as f:
            pickle.dump(self.clf,f)
    def loadModel(self):
        with open('svm.p','rb') as f:
            self.clf = pickle.load(f)

    def loadImgs(self):
        imageDirectory = self.image_dir
        with open(imageDirectory + 'train.txt', 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
        train = np.array([np.array(cv2.imread(imageDirectory +lines[i][0]+self.suffix)) for i in range(len(lines))])
        # read in training labels
        train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])
        self.train_imgs = train
        self.train_labels = train_labels

        imageDirectory = './2022Simgs/'
        self.suffix = '.jpg'
        with open(imageDirectory + 'test.txt', 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
        test = np.array([np.array(cv2.imread(imageDirectory +lines[i][0]+self.suffix)) for i in range(len(lines))])
        # read in testing labels
        test_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])
        self.test_imgs = test
        self.test_labels = test_labels
        return

    def getAccuracy(self,csv_filename):
        ### Run test images
        imageDirectory = self.image_dir
        with open(imageDirectory + csv_filename, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)

        correct = 0.0
        confusion_matrix = np.zeros((6,6))

        k = 21
        print('calculating confusion matrix')

        for i in range(len(lines)):
            original_img = cv2.imread(imageDirectory+lines[i][0]+self.suffix)

            test_label = np.int32(lines[i][1])
            #ret, results, neighbours, dist = knn.findNearest(test_img, k)
            self.actual_label = test_label
            ret = self.identify(original_img)

            if test_label == ret:
                print(str(lines[i][0]) + " Correct, " + str(ret))
                correct += 1
                confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
            else:
                confusion_matrix[test_label][np.int32(ret)] += 1
                
                print(f'{lines[i][0]}.png Wrong, {self.label2text[test_label]} classified as {self.label2text[ret]}')
            if(False and __debug__):
                cv2.imshow('debug', original_img)
                #cv2.imshow(Title_resized, test_img)
                key = cv2.waitKey()
                if key==27:    # Esc key to stop
                    break

        print("\n\nTotal accuracy: " + str(correct/len(lines)))
        print(confusion_matrix)
        return correct/len(lines)


    def getTrainingAccuracy(self):
        return self.getAccuracy('train.txt')

    def getTestAccuracy(self):
        return self.getAccuracy('test.txt')

    # identify an image
    def identify(self,ori_img):
        '''
        takes an image (loaded, BGR matrix, uint8), then give a label
        '''
        img,label = self.process(ori_img)
        return label


    def display(self,imgs,texts):
        count = len(imgs)
        if (count > 1):
            f, axarr = plt.subplots(1,count)
            for i in range(count):
                img = cv2.cvtColor(imgs[i],cv2.COLOR_BGR2RGB)
                axarr[i].imshow(img)
                axarr[i].title.set_text(texts[i])
        else:
            img = cv2.cvtColor(imgs[0],cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(texts[0])
        plt.show()

    # get the contour for a template
    def getContour(self,img,show=False):
        # First, check If there's a sign
        # If yes, then ...
        # remove background 
        # >120 in all channel is whie wall
        mask_b = img[:,:,0] > 100
        mask_g = img[:,:,1] > 100
        mask_r = img[:,:,2] > 100
        mask = np.bitwise_and(mask_b,mask_g)
        mask = np.bitwise_and(mask,mask_r)
        whites = np.sum(mask)
        total_pix = mask.shape[0] * mask.shape[1]
        white_ratio = whites/total_pix

        mask = mask.astype(np.uint8)
        kernel = np.ones((10,10),np.uint8)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        output = cv2.connectedComponentsWithStats(mask, 4)
        (numLabels, labels, stats, centroids) = output
        # stats: cv2.CC_STAT_LEFT, TOP, WIDTH, HEIGHT, AREA
        areas = [stat[cv2.CC_STAT_AREA] for stat in stats]
        wall_index = np.argmax(areas)
        mask_wall = labels == wall_index

        mask_wall = mask_wall.astype(np.uint8)
        contours, hier = cv2.findContours(mask_wall,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # find the largest contour (wall)
        areas = [cv2.contourArea(contour) for contour in contours]
        wall_idx = np.argmax(areas)
        # then find its biggest child
        hier = hier[0]
        i = hier[wall_idx][2] # 2: first child
        idx = i
        area = cv2.contourArea(contours[idx])
        while (hier[i][0] != -1): # 0: next
            i = hier[i][0]
            if (cv2.contourArea(contours[i]) > area):
                idx = i
                area = cv2.contourArea(contours[i])

        # NOTE found shape
        # contours[idx] is the sign
        cnt = contours[idx]
        if (show):
            blank = np.zeros_like(mask).astype(np.uint8)
            debug_img = cv2.drawContours(blank,[cnt],0,255,cv2.FILLED)
            plt.imshow(debug_img)
            plt.show()
        return cnt

    # process an image, 
    # return debugimg,label
    def process(self,img):
        # First, check If there's a sign
        # If yes, then ...
        # remove background 
        # >120 in all channel is whie wall
        mask_b = img[:,:,0] > 100
        mask_g = img[:,:,1] > 100
        mask_r = img[:,:,2] > 100
        mask = np.bitwise_and(mask_b,mask_g)
        mask = np.bitwise_and(mask,mask_r)
        whites = np.sum(mask)
        total_pix = mask.shape[0] * mask.shape[1]

        mask = mask.astype(np.uint8)
        kernel = np.ones((10,10),np.uint8)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        output = cv2.connectedComponentsWithStats(mask, 4)
        (numLabels, labels, stats, centroids) = output
        # stats: cv2.CC_STAT_LEFT, TOP, WIDTH, HEIGHT, AREA
        areas = [stat[cv2.CC_STAT_AREA] for stat in stats]
        wall_index = np.argmax(areas)
        mask_wall = labels == wall_index

        mask_wall = mask_wall.astype(np.uint8)
        contours, hier = cv2.findContours(mask_wall,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # find the largest contour (wall)
        areas = [cv2.contourArea(contour) for contour in contours]
        wall_idx = np.argmax(areas)
        # then find its biggest child
        hier = hier[0]
        i = hier[wall_idx][2] # 2: first child
        idx = i
        area = cv2.contourArea(contours[idx])
        while (hier[i][0] != -1): # 0: next
            i = hier[i][0]
            if (cv2.contourArea(contours[i]) > area):
                idx = i
                area = cv2.contourArea(contours[i])

        # NOTE found shape
        # contours[idx] is the sign
        cnt = contours[idx]
        x,y,w,h = cv2.boundingRect(cnt)

        blank = np.zeros(img.shape[:2],dtype=np.uint8)
        cnt_img = cv2.drawContours(blank,[cnt],0,255,-1)
        #ratio = img.shape[0]/h
        crop_img = cnt_img[y:y+h,x:x+w]
        resize_img = cv2.resize(crop_img, (30,30))

        img_area = img.shape[0]*img.shape[1]
        area = w*h

        #print(f'aspecr ratio: {w/h}')
        if (w/h > 3 or h/w>3):
            return None
        if (area/img_area < 0.01 or area/img_area>0.9):
            return None

        return resize_img



    # get all index of a label
    def getIndices(self, label):
        indices = np.where(self.train_labels == label)
        return indices

    # randomly test a label in training set
    def random(self):
        count = len(self.train_labels)
        while True:
            i = np.random.randint(0,count)
            self.debug([i])

    def train(self):
        training_data = []
        training_labels = []
        for (img, label) in zip(self.train_imgs, self.train_labels):
            processed = self.process(img)
            if (not processed is None):
                training_data.append(processed.flatten()/255-0.5)
                training_labels.append(label)

        clf = svm.SVC()
        clf.fit(training_data, training_labels)
        self.clf = clf

        correct_count = 0
        trained_labels = clf.predict(training_data)
        errors = len(np.nonzero(training_labels - trained_labels)[0])
        accuracy = 1-errors/len(trained_labels)
        print(f'training accuracy = {accuracy}')
        return

    def test(self):
        test_data = []
        test_labels = []
        for (img, label) in zip(self.test_imgs, self.test_labels):
            processed = self.process(img)
            if (not processed is None):
                test_data.append(processed.flatten()/255-0.5)
                test_labels.append(label)

        correct_count = 0
        predicted_labels = self.clf.predict(test_data)
        errors = len(np.nonzero(predicted_labels - test_labels)[0])
        accuracy = 1-errors/len(test_labels)
        print(f'test accuracy = {accuracy}')
        return

    def predict(self,img):
        processed = self.process(img).flatten()/255-0.5
        if (processed is None):
            return 0 # empmty
        else:
            return self.clf.predict([processed])[0]
        

if __name__=='__main__':
    main = Signs()
    #main.loadImgs()
    #main.train()
    #main.saveModel()
    #main.test()
    main.loadImgs()
    main.loadModel()
    label = main.predict(main.train_imgs[0])
    true_label = main.train_labels[0]
    print(label,true_label)

