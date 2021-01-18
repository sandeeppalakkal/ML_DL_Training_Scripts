import torch
import torchvision
import time
import torch.nn.functional as F


class LossFunctionL1(object):

    def __init__(self, args, device):

        if (torch.cuda.is_available()):
            self.L1LossFn = torch.nn.L1Loss().to(device)
        else:
            self.L1LossFn = torch.nn.L1Loss()

        self.L1lossVal = 0.0
        self.TotallossVal = 0.0

    def ResetLosses(self):

        self.L1lossVal = 0.0
        self.TotallossVal = 0.0

    def ComputeLosses(self, outIm, tIm):

        self.lossL1 = self.L1LossFn(outIm, tIm)

    def UpdateLosses(self):

        self.L1lossVal = self.L1lossVal + self.lossL1.item()
        self.TotallossVal = self.L1lossVal


    def Backward(self):

        self.lossL1.backward()


    def PrintLosses(self, pTag, iter):

        print(pTag + "L1: " + str(self.TotallossVal/iter))



def GetLossFunction(args, device):

    return LossFunctionL1(args, device)



