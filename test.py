#!/usr/bin/env python3

import sys
from argparse import ArgumentParser
import torch
import torch.backends.cudnn as cudnn

IMPLEMENTED_TRAIN_ENGINE = [
    "DPIG-Pose",
]


def select_test(name):
    if name == IMPLEMENTED_TRAIN_ENGINE[0]:
        from test import PoseSample as test
    # elif name == IMPLEMENTED_TRAIN_ENGINE[1]:
    #    pass
    else:
        raise NotImplementedError("You have not implement {}".format(name))
    return test


def main():
    top_parser = ArgumentParser(description='Testing', add_help=False)
    top_parser.add_argument("--name",
                            default=IMPLEMENTED_TRAIN_ENGINE[0],
                            type=str,
                            choices=IMPLEMENTED_TRAIN_ENGINE,
                            help="test what?")
    top_parser.add_argument("-h", "--help",
                            action='store_true',
                            help="""
                                show this message. if name is specified, 
                                help message for test arg parser also will be show.
                                """)
    top_opt, other_args = top_parser.parse_known_args()

    have_specified_name = "--name" in sys.argv

    if top_opt.help and not have_specified_name:
        top_parser.print_help()
        print("\nSupported test engine for now:\n")
        for e in IMPLEMENTED_TRAIN_ENGINE:
            print("* " + e)
        print("\tYou must choose one of above as engine!")
        return

    test = select_test(top_opt.name)

    parser = ArgumentParser(description='Testing {}'.format(top_opt.name), add_help=False)
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu_id: e.g. 0')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument("--output_dir", type=str, default="ckp/")
    test.add_new_arg_for_parser(parser)

    opt = parser.parse_args(other_args)

    if top_opt.help and have_specified_name:
        print(parser.format_help())
        return

    torch.cuda.set_device(opt.gpu_id)
    print("Begin to test {}, using GPU {}".format(top_opt.name, opt.gpu_id))
    cudnn.benchmark = True
    device = "cuda"

    test_engine = test.get_tester(opt, device)
    data_loader = test.get_data_loader(opt)
    test_engine.run(data_loader, max_epochs=1)


if __name__ == '__main__':
    main()