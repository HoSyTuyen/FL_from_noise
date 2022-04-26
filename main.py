#!/usr/bin/env python
import argparse
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverFedProx import FedProx
from FLAlgorithms.servers.serverFedDistill import FedDistill
from FLAlgorithms.servers.serverpFedGen import FedGen
from FLAlgorithms.servers.serverpFedEnsemble import FedEnsemble
from FLAlgorithms.servers.serverSP import FedSP
from utils.model_utils import create_model
from utils.plot_utils import *
import torch
from multiprocessing import Pool

from torch.utils.tensorboard import SummaryWriter

def create_server_n_user(args, i):
    model = create_model(args.model, args.dataset, args.algorithm)
    if ('FedAvg' in args.algorithm):
        server=FedAvg(args, model, i)
    elif 'FedGen' in args.algorithm:
        server=FedGen(args, model, i)
    elif ('FedProx' in args.algorithm):
        server = FedProx(args, model, i)
    elif ('FedDistill' in args.algorithm):
        server = FedDistill(args, model, i)
    elif ('FedEnsemble' in args.algorithm):
        server = FedEnsemble(args, model, i)
    elif ('FedSP' in args.algorithm):
        server = FedSP(args, model, i)
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server


def run_job(args, i):
    torch.manual_seed(i)
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i)
    writer = SummaryWriter(comment=args.writer, log_dir="runs/{}".format(args.writer))
    if args.train:
        server.train(args, writer)
        server.test()

def main(args):
    for i in range(args.times):
        run_job(args, i)
    print("Finished training.")


if __name__ == "__main__":
    torch.manual_seed(1000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--algorithm", type=str, default="pFedMe")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate.")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    parser.add_argument("--num_glob_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--num_users", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=3, help="running time")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")
    
    # KD server hyper-param
    parser.add_argument("--KD_epoch", type=int, default=20, help="# KD epoch")
    parser.add_argument("--KD_weight", type=int, default=4, help="KD weight")
    parser.add_argument("--KD_lr", type=float, default=0.001, help="KD lr")
    parser.add_argument("--KD_temperature", type=int, default=3, help="KD temperature")
    parser.add_argument("--KD_bs", type=int, default=4096, help="KD bs")
    parser.add_argument("--KD_latent_size", type=int, default=100, help="KD latent size")
    parser.add_argument("--KD_num_per_client", type=int, default=4096, help="KD number of generated image per client")
    parser.add_argument("--pretrained_G", default="FLAlgorithms/trainmodel/pretrained/G_emnist_200E.ckpt", type=str, help="path to pre-trained Generator, None if no pre-trained")
    
    parser.add_argument("--KD_selective_feature_ratio", type=float, default=1.0, help="selective logits for KD")
    parser.add_argument("--train_with_img", default=False, type=bool, help="debug mode to use real image")
    parser.add_argument("--track_KL", default=False, type=bool, help="track KL mode")

    parser.add_argument("--writer", type=str, default="real_images_track_KL", help="directory path to writer")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Ensemble learing rate       : {}".format(args.ensemble_lr))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Device            : {}".format(args.device))
    print("=" * 80)
    main(args)
