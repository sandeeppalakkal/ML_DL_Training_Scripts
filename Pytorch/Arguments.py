import os
import argparse

def SetupArguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inputDir', type=str, default="./", help='Input Data Directory')
    parser.add_argument('--resultsDir', type=str, default="./", help='Output Results Directory')
    parser.add_argument('--ckptDir', type=str, default="./", help='Output Checkpoints Directory')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch Size')
    parser.add_argument('--gpuID', type=int, default=0, help='GPU ID')
    parser.add_argument('--numEpochs', type=int, default=100, help='Total no of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
    parser.add_argument('--weightDecay', type=float, default=0.005, help='Weight Decay')
    parser.add_argument('--ckptFile', type=str, default="./ckpt.pt", help='Checkpoint File')
    parser.add_argument('--transform', default=False, action='store_true', help='Enable Data transformation')
    parser.add_argument('--fineTuning', default=False, action='store_true', help='For Transfer Learning')
    parser.add_argument('--evalFreq', type=int, default=10, help='Eval Frequency')
    parser.add_argument('--saveFreq', type=int, default=10, help='Output Checkpoints save Frequency')

    args = parser.parse_args()

    return args

