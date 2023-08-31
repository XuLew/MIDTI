import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run ReGCN.")
    parser.add_argument("--name",
                        nargs="?",
                        default="protein",
                        help="Training similarity datasets name，drug or protein")

    parser.add_argument("--pretrain-dataset-path",
                        nargs="?",
                        default="../../dataset/LuoDTI/used_data/",
                        help="PreTraining datasets.")

    parser.add_argument("--train-dataset-path",
                        nargs="?",
                        default="../../dataset/LuoDTI/input/",
                        help="Training datasets.")

    parser.add_argument("--epoch",
                        type=int,
                        default=1000,
                        help="Number of training epochs. Default is 1000.")
    
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Rate of learning. Default is 0.001.")

    parser.add_argument("--gcn-layers",
                        type=int,
                        default=2,
                        help="Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--d-out-channels",
                        type=int,
                        default=708,
                        help="out-channels of cnn/the dim of drug output features. Default is 708")

    parser.add_argument("--p-out-channels",
                        type=int,
                        default=1512,
                        help="out-channels of cnn/the dim of protein output features. Default is 1512.")

    parser.add_argument("--drug-number",
                        type=int,
                        default=708,
                        help="drug number. Default is 708.")

    parser.add_argument("--f",
                        type=int,
                        default=512,
                        help="dim. Default is 512.")

    parser.add_argument("--protein-number",
                        type=int,
                        default=1512,
                        help="miRNA number. Default is 1512.")


    parser.add_argument("--view-d",
                        type=int,
                        default=5,
                        help="views number. Default is 5(5 datasets for drug sim)")

    parser.add_argument("--view-p",
                        type=int,
                        default=4,
                        help="views number. Default is 4(4 datasets for protein sim)")


    return parser.parse_args()