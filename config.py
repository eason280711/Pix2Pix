import argparse
from calendar import EPOCH

LR = 2e-4
BATCH_SIZE = 32
L1_LAMBDA = 100
EPOCHS = 1000
NUM_WORKERS = 6
DATA_PATH = "../data"


parser = argparse.ArgumentParser(prog="config.py", description="Config")
parser.add_argument("--path", "-p", default="../data", type=str, required=False,  help="Path to Dataset")
parser.add_argument("--epochs", "-e", default=1000, type=int, required=False,  help="how much epochs")
parser.add_argument("--l1lambda", "-l1", default=100, type=int, required=False,  help="L1 lambda")
parser.add_argument("--batchsize", "-b", default=32, type=int, required=False,  help="Batch size")
parser.add_argument("--learningrate", "-lr", default=2e-4, type=int, required=False,  help="Learing Rate")
parser.add_argument("--workers", "-w", default=6, type=int, required=False,  help="nums workers")
args = parser.parse_args()

LR = args.learningrate
BATCH_SIZE = args.batchsize
L1_LAMBDA = args.l1lambda
EPOCHS = args.epochs
DATA_PATH = args.path
NUM_WORKERS = args.workers