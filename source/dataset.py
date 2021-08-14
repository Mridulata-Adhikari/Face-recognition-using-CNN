import numpy as np
import cv2
import os
import re
import pickle

from imutils import paths

class GetDataSet:

    # Load the face data
    N_POSES = 100
    N_SUBJECTS = 5
    HEIGHT = 112
    WIDTH = 92

    def read_jpg(self, filename):
        print(filename)
        im = cv2.imread(filename)
        image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # image = cv2.imread(filename, 0)

        return image

    def selector_generator(self, num):  # Selector for the poses present in training and testing sets
        poses = list(range(self.N_POSES))  # pose indexes i.e [0 1 2 3 .. 8 9]
        print("total no of poses: ", len(poses))

        np.random.shuffle(poses)  # Randomly arrange the pose indexes

        num_trainposes = num + 1
        num_testposes = 1

        train_select = poses[0:num_trainposes]
        test_selector = poses[num_trainposes:]

        num_t = num_trainposes - num_testposes

        np.random.shuffle(poses)  # Randomly arrange the pose indexes
        train_selector = train_select[0:num_t]
        valid_selector = train_select[num_t:]

        print(train_selector)
        print(valid_selector)
        print(test_selector)

        return train_selector, valid_selector, test_selector



    def dataset(self,num):
        address = "images/train/custom_dataset/"
        train_selector, valid_selector, test_selector = self.selector_generator(num)

        # initially our image is of size 112 X 92 pixels
        trainset = np.ones((self.N_SUBJECTS * len(train_selector), 1, self.HEIGHT,
                            self.WIDTH))  # Image_Number X Channel X Image_Height X Image_Width
        validset = np.ones((self.N_SUBJECTS * len(valid_selector), 1, self.HEIGHT, self.WIDTH))
        testset = np.ones((self.N_SUBJECTS * len(test_selector), 1, self.HEIGHT, self.WIDTH))

        train_labels = np.ones((self.N_SUBJECTS * len(train_selector), 1))
        valid_labels = np.ones((self.N_SUBJECTS * len(valid_selector), 1))
        test_labels = np.ones((self.N_SUBJECTS * len(test_selector), 1))

        train_counter = 0
        valid_counter = 0
        test_counter = 0

        names_dict = ["mridulata", "saloni", "smrity", "sujata", "unknown"]
        train_no = 0

        # Generate Train and Test Sets
        for subject in range(1, 5 + 1):
            for pose in train_selector:
                image = self.read_jpg(address + str(names_dict[train_no])+ "/" + str(names_dict[train_no])+ str(pose) + ".jpg")
                trainset[train_counter, 0, :, :] = image
                train_labels[train_counter, 0] = subject - 1
                train_counter += 1

            for pose in valid_selector:
                image = self.read_jpg(address + str(names_dict[train_no])+ "/" + str(names_dict[train_no])+ str(pose) + ".jpg")
                validset[valid_counter, 0, :, :] = image
                valid_labels[valid_counter, 0] = subject - 1
                valid_counter += 1

            for pose in test_selector:
                image = self.read_jpg(address + str(names_dict[train_no])+ "/" + str(names_dict[train_no])+ str(pose) + ".jpg")
                testset[test_counter, 0, :, :] = image
                test_labels[test_counter, 0] = subject - 1
                test_counter += 1

        train_no+= 1
        return trainset, train_labels, validset, valid_labels, testset, test_labels





