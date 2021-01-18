import os
import sys
import torch
import math
import random
from os.path import join
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import scipy.stats as st
import cv2

to_tensor = transforms.ToTensor()


class SingleImageDataset(Dataset):

    def __init__(self, datasetType, args):

        self.args = args
        self.datasetType = datasetType

        self.transform   = self.args.transform
        self.isTraining  = False
        self.isTesting   = False
        self.isGTPresent = True
        inputDir         = args.inputDir

        if datasetType == "TrainSingleImage":
            inputDir   = join(inputDir, "train")
            inputDir_b = join(inputDir, "input")
            self.isTraining = True

        if datasetType == "EvalSingleImage":
            inputDir   = join(inputDir, "eval")
            inputDir_b = join(inputDir, "input")

        if datasetType == "TestSingleImage":
            inputDir_b = join(inputDir, "input")
            self.isTesting = True
            self.isGTPresent = False

        if datasetType == "TestSingleImageWithLoss":
            inputDir_b = join(inputDir, "input")
            self.isTesting = True

        if self.isGTPresent:
            inputDir_t = join(inputDir, "groundTruth")
            fileList_t = []

        fileList_b = []

        dirList = os.listdir(inputDir_b)

        for file in dirList:
            fullfilepath_b = join(inputDir_b, file)
            if self.isGTPresent:
                fullfilepath_t = join(inputDir_t, file)
                fileList_t.append(fullfilepath_t)
            fileList_b.append(fullfilepath_b)

        self.datasetLength = len(fileList_b)
        if self.isGTPresent:
            self.fileList_t = fileList_t
        self.fileList_b = fileList_b


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_b = Image.open(self.fileList_b[idx])
        if self.isGTPresent:
            img_t = Image.open(self.fileList_t[idx])

        if self.transform:
            scalingFactor = 1.0
            if self.isTraining:
                img_t = self.ResizeData(img_t, scalingFactor)
                img_b = self.ResizeData(img_b, scalingFactor)
                img_b, img_t = self.PairedTransform(img_b, img_t)
            else:
                img_b = self.ResizeData(img_b, scalingFactor)
                if isGTPresent:
                    img_t = self.ResizeData(img_t, scalingFactor)

        T_img_b = to_tensor(img_b)
        fname_b = os.path.basename(self.fileList_b[idx])
        if self.isGTPresent:
            T_img_t = to_tensor(img_t)
            fname_t = os.path.basename(self.fileList_t[idx])
            sample  = {"Input": T_img_b, "GroundTruth": T_img_t, "DatasetType": self.datasetType, "FileName_b": fname_b}
        else:
            sample  = {"Input": T_img_b, "DatasetType": self.datasetType, "FileName_b": fname_b}
        return sample


    def __len__(self):
        return self.datasetLength

    def PairedTransform(self, img1, img2):

        imgWidth, imgHeight = img1.size

        patchWidth  = 256
        patchHeight = 256

        i = random.randint(0, imgHeight - patchHeight)
        j = random.randint(0, imgWidth  - patchWidth)

        if random.random() < 0.5:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)

        img1 = F.crop(img1, i, j, patchHeight, patchWidth)
        img2 = F.crop(img2, i, j, patchHeight, patchWidth)

        return img1, img2

    def ResizeData(self, img, scalingFactor=1):

        w,h = img.size

        newH = (int(h / scalingFactor) + 31) & (-32)
        newW = (int(w / scalingFactor) + 31) & (-32)

        img = img.resize((newH, newW), Image.BICUBIC)

        return img





