import os
import DataHandler
import System
import Arguments

def main():

    args = Arguments.SetupArguments()

    PrintArgs(args)

    if not os.path.exists(args.resultsDir):
        os.makedirs(args.resultsDir)
    if not os.path.exists(args.resultsDir):
        os.makedirs(args.resultsDir)


    pSystem    = System.SingleImageSystem(args)
    pTrainData = DataHandler.SingleImageDataset("TrainSingleImage", args)
    pEvalData  = DataHandler.SingleImageDataset("TrainSingleImage", args)

    pSystem.StartTraining(pTrainData, pEvalData)


def PrintArgs(obj):

    print("\n ==================Options================")
    for k,v in obj,__dict__.items():
        print(str(k) + "=" + str(v))
        if hasattr(v, '__dict__'):
            PrintArgs(v)
    print("\n ==================Options================\n")


if __name__ == '__main__':
    main()

