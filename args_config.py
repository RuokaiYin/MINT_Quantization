import argparse


def get_args():

    parser = argparse.ArgumentParser("UQSNN")

    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--dump-dir', type=str, default="logdir")
    # parser.add_argument("--encode", default="d", type=str, help="Encoding [p d]")
    parser.add_argument("--arch", default="res19", type=str, help="Arch [vgg9,vgg16,res19]")
    parser.add_argument('--dataset_dir', type=str, default='../dataset/', help='path to the dataset')    
    parser.add_argument("--dataset", default="dvs", type=str, help="Dataset [cifar10,svhn,tiny,dvs]")
    parser.add_argument("--optim", default='adam', type=str, help="Optimizer [adam, sgd]")
    parser.add_argument('--leak_mem',default=0.5, type=float)
    parser.add_argument('--th',default=0.5, type=float)
    parser.add_argument('--rst',default="hard", type=str)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('-uq', action='store_true')
    parser.add_argument('-bq', action='store_true')
    parser.add_argument('-wq', action='store_true')
    parser.add_argument('-share', action='store_true')
    parser.add_argument('-sft_rst', action='store_true')
    parser.add_argument('-conv_b', action='store_true')
    parser.add_argument('-bn_a', action='store_true')
    parser.add_argument('-xa', action='store_true')
    parser.add_argument('-ts', action='store_true')

    parser.add_argument('--epoch', type=int, default=200)
    # parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    parser.add_argument("--train_display_freq", default=1, type=int, help="display_freq for train")
    parser.add_argument("--test_display_freq", default=1, type=int, help="display_freq for test")
    # parser.add_argument("--setting", type=str, help="display_freq for test")
    # parser.add_argument('--quant',     default=4, type=int, help='quantization-bits')
    args = parser.parse_args()

    

    return args