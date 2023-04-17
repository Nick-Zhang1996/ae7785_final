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

    def prepareTemplateContours(self):
        # left curve, left, right, right, curve, stop, goal, return
        indices = [10,13,48,54,66,99,109]
        self.lookup = lookup = ['left curve',' left',' right','right curve','stop',' goal','return']
        imgs = np.array([np.array(cv2.imread(f'{self.image_dir}{i}.png')) for i in indices])
        self.template_contours = cnts = [self.getContour(img) for img in imgs]
        return

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

    # identify contour to label
    def identifyContour(self,cnt,img):
        # calculate matching score for each template
        scores = [cv2.matchShapes(cnt,cnt_candidate,1,0.0) for cnt_candidate in self.template_contours]
        scores = np.array(scores)
        # confidence
        exp_scores = np.exp(-scores)
        exp_scores = exp_scores/np.sum(exp_scores)

        # most compatible shape
        # shape index: left curve, left, right, right curve, stop, goal, return
        shape2label = [1,1,2,2,4,5,3]
        shape = np.argmin(scores)
        confidence = exp_scores[shape]
        if (confidence > 0.29):
            pass
            #breakpoint()
            #print(f'shape confidence {confidence}')

        # matchShapes is good at finding stop and goal, but can't distinguish arrow very well
        if (shape in (1,2)):
            return self.LeftOrRight(cnt,img)
        elif (shape in (4,5)):
            # map shape to actual label
            return shape2label[shape]
        else:
            return self.identifyChevron(cnt,img)

        # should never reach here
        return 0




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



    def identifyChevron(self, cnt,img):
        # identify circles
        blank = np.zeros(img.shape[:2],dtype=np.uint8)
        epsilon = 0.02*cv2.arcLength(cnt,True)
        cnt_img = cv2.drawContours(blank,[cnt],0,255,1)
        x,y,w,h = cv2.boundingRect(cnt)
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        ratio = img.shape[0]/h
        crop_img = cnt_img[y:y+h,x:x+w]
        resize_img = cv2.resize(crop_img, (int(w*ratio), int(h*ratio)))


        circles = cv2.HoughCircles(resize_img,cv2.HOUGH_GRADIENT,1,20,
                            param1=30,param2=20,minRadius=0,maxRadius=0)
        if (circles is None):
            return 0
        circles = np.uint16(np.around(circles))

        debug_img = np.dstack([resize_img]*3)

        i = circle = circles[0,0]

        label = -1
        # if circle center is near bottom, then it's a left/right
        #print(f'circle vertical {circle[1]/img.shape[0]}')
        #print(f'circle lateral (l-r) {circle[0]/img.shape[1]}')
        if (circle[1] > 0.7*img.shape[0]):
            # TODO check left or right
            if ( circle[0] < 0.5*img.shape[1]):
                label = 1 #left
            else:
                label = 2 #right
        else:
            label = 3 #do not enter

        '''
        if (label != self.actual_label):
            print(f'{self.actual_label} ->{label}')
            breakpoint()
        '''
        return label


        # draw the outer circle
        cv2.circle(debug_img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(debug_img,(i[0],i[1]),2,(0,0,255),3)

        plt.imshow(debug_img)
        plt.show()


        return 0

    def LeftOrRight(self, cnt, img):
        blank = np.zeros(img.shape[:2],dtype=np.uint8)
        epsilon = 0.02*cv2.arcLength(cnt,True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
        cnt_img = cv2.drawContours(blank,[cnt],0,255,1)
        # img, rho, theta, threshold, min_len, max_gap
        lines = cv2.HoughLinesP(cnt_img, 1, np.pi/180, 20, None, 15, 15)
        # is there a long vertical line?

        l_len_vec = []
        l_vec = []
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                if (l[2] == l[0]):
                    l_slope = 100
                else:
                    l_slope = np.abs((l[3]-l[1])/(l[2]-l[0]))
                l_len = ((l[3]-l[1])**2 + (l[2]-l[0])**2)**0.5
                if (l_slope > 3):
                    l_vec.append(l)
                    l_len_vec.append(l_len)
            if (len(l_vec) == 0):
                #print(f'cant find vertical line')
                '''
                img = cv2.drawContours(img,[cnt],0,(0,0,0),1)
                for i in range(0, len(lines)):
                    l = lines[i][0]
                    img = cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
                plt.imshow(img)
                plt.show()
                breakpoint()
                '''
                return 0
            l = longest_vertical_l = l_vec[np.argmax(l_len_vec)]
            l_center_x = (l[2]+l[0])/2
            M = cv2.moments(cnt)
            # FIXME height*width
            # top left: origin
            try:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            except ZeroDivisionError:
                return 0
            #print(f'cx: {cx}')
            #print(f'l_center_x: {l_center_x}')

            if (cx < l_center_x):
                label = 1
            else:
                label = 2
            return label
            

        return 0


    # test and display a set of images
    def debug(self, indices=(25,0,52,8,64,97)):
        imgs = []
        processed_imgs = []
        texts = []
        for i in indices:
            img = self.train_imgs[i]
            text = f'{i}: true label: {self.label2text[self.train_labels[i]]}'
            print(' -------------- ')
            print(text)
            processed_img, label = self.process(img)
            print(f'identified as :{self.label2text[label]}')
            imgs.append(img)
            processed_imgs.append(processed_img)
            texts.append(text)
        self.display(processed_imgs,texts)

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
        

if __name__=='__main__':
    main = Signs()
    main.loadImgs()
    main.train()
    main.saveModel()
    main.test()

