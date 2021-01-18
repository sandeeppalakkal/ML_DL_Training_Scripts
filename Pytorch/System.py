import os
import sys
from os.path import join
import torch
import torch.optim as optim
import torch.nn as nn
import Network as net
from datetime import datetime
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f
import Losses
import math
import time

def weights_init(m):

    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)



class SingleImageSystem(object):

    def __init__(self, args):
        self.args = args

    def Initialize(self, args):
        print("Initializing System")

    def StartTesting(self, test_data):
        print("Testing")

        batchSize = self.args.batchsize

        if torch.cuda.is_available():
            device = torch.device(self.args.gpuID)
        else:
            device = torch.device("cpu")

        tDataLoader = DataLoader(test_data, batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)

        network = self.GetNetwork()

        if torch.cuda.device_count() > 1:
            print("Enabling Data parallel")
            network = nn.DataParallel(network)


        network.to(device)
        self.network = network

        epoch, loss = self.LoadModel(self.args.ckptFile, False)

        if(self.args.gpuID == -1):
            network.cpu()
        else:
            network.cuda(self.args.gpuID)

        self.loss_fns = Losses.GetLossFunction(self.args, device)
        self.loss_fns.ResetLosses()

        iter = 0
        tot = len(test_data)

        for batchIdx, item in enumerate(tDataLoader):

            inIm = item["Input"]
            tIm  = item["GroundTruth"]

            if torch.cuda.is_available():
                with torch.no_grad():
                    inIm = inIm.to(device)
                    tIm  = tIm.to(device)

            outIm = network.forward(inIm)

            iter += 1
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            pTag = "[" + current_time + "] Processing Item = " + str(batchIdx) + "/" + str(tot//batchSize) + ", Testing Losses:"
            self.loss_fns.PrintLosses(pTag, iter)

            imName1 = os.path.splitext(imName[idx])[0]
            fname_t = imName1 + "_input.jpg"
            fname_t = join(self.args.resultsDir, fname_t)
            inImage = self.GetImageFromTensor(inIm)
            inImage.save(fname_t)

            fname_t = imName1 + "_output.jpg"
            fname_t = join(self.args.resultsDir, fname_t)
            outIm   = self.GetImageFromTensor(outIm)
            outIm.save(fname_t)

       now = datetime.now()
       current_time = now.strftime("%H:%M:%S")
       pTag = "\nTesting Complete[" + current_time + "] Processing Item = " + str(batchIdx) + "/" + str(tot//batchSize) + ", Testing Losses:"
       self.loss_fns.PrintLosses(pTag, iter)


    def StartTraining(self, train_data, eval_data):

        print("Start Training")

        if torch.cuda.is_available():
            device = torch.device(self.args.gpuID)
        else:
            device = torch.device("cpu")


        startepoch = 0
        lrate = self.args.lr
        wtDecay = selgs.args.weightDecay

        network = GetNetwork()


        if torch.cuda.device_count() > 1:
            print("Enabling Data parallel")
            network = nn.DataParallel(network)


        network.apply(weights_init)
        network.to(device)

        batchSize = self.args.batchsize

        tDataLoader = DataLoader(train_data, batch_size=batchSize, shuffle=False, num_workers=8, pin_memory=True)

        optimizer = optim.Adan(network.parameters(), lr=lrate, weight_decay=wtDecay)
        self.network = network
        self.optimizer = optimizer

        if(self.args.fineTuning):
            print("Loading from existing check point")
            startepoch, loss = self.LoadModel(self.args.ckptFile, True)
            network.train()

        self.loss_fns = Losses.GetLossFunction(self.args.device)

        for epoch in range(self.args.numEpochs):

            print('Epoch: {}\n'.format(startepoch + epoch + 1))

            self.loss_fns.ResetLosses()
            iter = 0
            tot = len(train_data)

            for batchIdx, item in enumerate(tDataLoader):

                inIm = item["Input"]
                tIm  = item["GroundTruth"]

                if torch.cuda.is_available():
                    inIm = inIm.to(device)
                    tIm  = tIm.to(device)

                optimizer.zero_grad()
                outIm = network.forward(inIm)
                self.loss_fns.ComputeLosses(outIm, tIm)
                self.loss_fns.Backward()
                optimizer.step()
                self.loss_fns.UpdateLosses()

                iter += 1
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                pTag = "[" + current_time + "] Processing Item = " + str(batchIdx) + "/" + str(tot//batchSize) + ", Testing Losses:"
                self.loss_fns.PrintLosses(pTag, iter)

            if(epoch % self.args.evalFreq == 0):
                self.EvalModel(eval_data, startepoch + epoch)

            if(epoch % self.args.saveFreq == 0):
                self.SaveModel(startepoch + epoch, self.loss_fns.TotallossVal)



    def LoadModel(self, ckptFile, isTraining):

        ckpt = torch.load(ckptFile)
        self.network.load_state_dict(ckpt['model_state_dict'])
        print("\n Loading Network Weights - Done")
        if isTraining:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("\n Loading Optimizer parameters and buffers - Done")
        epoch = ckpt['epoch']
        loss  = ckpt['loss']
        return epoch, loss


    def SaveModel(self, epoch, loss):
        ckptFname = "chkpt_" + str(epoch) + ".pt"
        torch.save({'epoch': epoch, 'model_state_dict': self.network.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'loss': loss, }, join(self.args.ckptDir, ckptFname))


    def GetImageFromTensor(self, im):

        im1 = im.cpu()
        n,c,h,w = im1.shape
        im1 = im1.reshape([c,h,w])
        im1 = torch.clamp(im1, 0.0, 1.0)*255.0
        im1 = im1.detach().numpy()
        im1 = im1.transpose((1,2,0))
        out = transforms.ToPILImage()(im1.astype(np.uint8))
        return out


    def GetNetwork(self):
        args = self.args
        network = net.Network(3,3, True)
        return network


    def EvalModel(self, eval_data, epoch):

        eDataLoader   = DataLoader(train_data, batch_size=batchSize, shuffle=False, num_workers=8, pin_memory=True)
        self.loss_fns = Losses.GetLossFunction(self.args, device)
        self.loss_fns.ResetLosses()

        iter = 0
        for item in eDataLoader:

            inIm = item["Input"]
            tIm  = item["GroundTruth"]

            if torch.cuda.is_available():
                inIm = inIm.to(device)
                tIm  = tIm.to(device)

            with torch.no_grad():
                outIm = network.forward(inIm)
                self.loss_fns.ComputeLosses(outIm, tIm)

            iter += 1
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            pTag = "[" + current_time + "] Processing Item = " + str(batchIdx) + "/" + str(tot//batchSize) + ", Testing Losses:"
            self.loss_fns.PrintLosses(pTag, iter)

            imName  = item["FileName_b"]
            imName1 = os.path.splitext(imName[idx])[0]
            fname_t = imName1 + "_input.jpg"
            fname_t = join(self.args.resultsDir, fname_t)
            inImage = self.GetImageFromTensor(inIm)
            inImage.save(fname_t)

            fname_t = imName1 + "_output.jpg"
            fname_t = join(self.args.resultsDir, fname_t)
            outIm   = self.GetImageFromTensor(outIm)
            outIm.save(fname_t)

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            pTag = "[" + current_time + "] Processing Item = " + str(batchIdx) + "/" + str(tot//batchSize) + ", Testing Losses:"
            self.loss_fns.PrintLosses(pTag, iter)


       now = datetime.now()
       current_time = now.strftime("%H:%M:%S")
       pTag = "\nTesting Complete[" + current_time + "] Processing Item = " + str(batchIdx) + "/" + str(tot//batchSize) + ", Testing Losses:"
       self.loss_fns.PrintLosses(pTag, iter)





