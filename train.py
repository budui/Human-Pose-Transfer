#!/usr/bin/env python3

import sys
from argparse import ArgumentParser
import torch
import torch.backends.cudnn as cudnn

IMPLEMENTED_TRAIN_ENGINE = [
    "DPIG-1-Pose",
    "PG2-2",
    "PG2-1"
]


def select_train(name):
    if name == IMPLEMENTED_TRAIN_ENGINE[0]:
        from train.DPIG import pose_auto_encoder as train
    elif name == IMPLEMENTED_TRAIN_ENGINE[1]:
        from train.PG2 import stage_2 as train
    elif name == IMPLEMENTED_TRAIN_ENGINE[2]:
        from train.PG2 import stage_1 as train
    else:
        raise NotImplementedError("You have not implement {}".format(name))
    return train


def main():
    top_parser = ArgumentParser(description='Training', add_help=False)
    top_parser.add_argument("--name",
                            default=IMPLEMENTED_TRAIN_ENGINE[0],
                            type=str,
                            choices=IMPLEMENTED_TRAIN_ENGINE,
                            help="train what?")
    top_parser.add_argument("-h", "--help",
                            action='store_true',
                            help="""
                                show this message. if name is specified, 
                                help message for train arg parser also will be show.
                                """)
    top_opt, other_args = top_parser.parse_known_args()

    have_specified_name = "--name" in sys.argv

    if top_opt.help and not have_specified_name:
        top_parser.print_help()
        print("\nSupported train engine for now:\n")
        for e in IMPLEMENTED_TRAIN_ENGINE:
            print("* " + e)
        print("\tYou must choose one of above as engine!")
        return

    train = select_train(top_opt.name)

    parser = ArgumentParser(description='Training {}'.format(top_opt.name))
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu_id: e.g. 0')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument("--epochs", default=20, type=int, help="epoch_num")
    parser.add_argument("--output_dir", type=str, default="ckp/")
    parser.add_argument("--print_freq", type=int, default=100, help="freq of print log message, "
                                                                    "default is every 100 iterations")
    parser.add_argument("--save_interval", default=1, type=int, help="models will be saved to disk "
                                                                     "every save_interval calls to the handler.")
    parser.add_argument("--n_saved", default=10, type=int, help="Number of models that should be kept on disk. "
                                                                "Older files will be removed.")
    train.add_new_arg_for_parser(parser)

    opt = parser.parse_args(other_args)

    if top_opt.help and have_specified_name:
        print(parser.format_help())
        return

    torch.cuda.set_device(opt.gpu_id)
    print("Begin to train {}, using GPU {}".format(top_opt.name, opt.gpu_id))
    cudnn.benchmark = True
    device = "cuda"

    train_engine = train.get_trainer(opt, device)
    data_loader = train.get_data_loader(opt)
    train_engine.run(data_loader, max_epochs=opt.epochs)


if __name__ == '__main__':
    main()