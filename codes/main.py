import argparse
# from dataset import APDDataset
from utils import setSeed, getDevice
from subprocess import call
# from train import train

config = {
    'seed': 87,
    'epoch': 200,
    'batch_size': 32,
    'optimizer': 'Adam',
    'optim_parameters': {
        'lr': 0.001,
        # weight_decay
    },
    # 'scheduler': ,
    # 'sched_parameters': {},
    'criterion': 'CrossEntropyLoss',
    'save_path': 'models/model.ckpt'
}

# parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--mode', dest='mode', type=str)
parser.add_argument('--seed', dest='seed', type=int, default=config['seed'])
parser.add_argument('--bs', dest='batch_size', type=int, default=config['batch_size'])
parser.add_argument('--epoch', dest='epoch', type=int, default=config['epoch'])
parser.add_argument('--lr', dest='lr', type=float, default=config['optim_parameters']['lr'])
parser.add_argument('--model', dest='model_name', type=str, default='sphereface')

if __name__ == '__main__':

    args = parser.parse_args()

    assert args.mode == 'train' or args.mode == 'close' or args.mode == 'open', \
           '[ERROR] Mode incorrect! Should be one of "train", "close", "open"!'

    # set random seed
    setSeed(args.seed)
    # set config
    config['device'] = getDevice()
    config['mode'] = args.mode

    if args.mode == 'train':
        # my NN
        # model = MyModel()
        # train(model, config)
        if args.model_name == 'sphereface':
            # -W igonre: ignore warning messages
            call("cd sphereface_pytorch; python3 -W ignore train.py", shell=True)
        elif args.model_name == 'sphereface_base':
            # -W ignore: ignore warning messages
            call("cd sphereface_CELoss; python3 -W ignore train.py", shell=True)
        elif args.model_name == 'sphereface_cos':
            call("cd sphereface_CosLoss; python3 -W ignore train.py --lr 0.1", shell=True)
        else:
            print(f'{args.model_name} doesn\'t exist!')

    elif args.mode == 'close':
        if args.model_name == 'sphereface':
            # -W ignore: ignore warning messages
            call("cd sphereface_pytorch; python3 -W ignore test.py --mode close", shell=True)
        elif args.model_name == 'sphereface_base':
            # -W ignore: ignore warning messages
            call("cd sphereface_CELoss; python3 -W ignore test.py --mode close", shell=True)
        elif args.model_name == 'sphereface_cos':
            call("cd sphereface_CosLoss; python3 -W ignore test.py --mode close", shell=True)

    elif args.mode == 'open':
        if args.model_name == 'sphereface':
            # -W ignore: ignore warning messages
            call("cd sphereface_pytorch; python3 -W ignore test.py --mode open", shell=True)
        elif args.model_name == 'sphereface_base':
            # -W ignore: ignore warning messages
            call("cd sphereface_CELoss; python3 -W ignore test.py --mode open", shell=True)
        elif args.model_name == 'sphereface_cos':
            call("cd sphereface_CosLoss; python3 -W ignore test.py --mode open", shell=True)
