from random import shuffle
# from typing import _get_type_hints_obj_allowed_types
from matplotlib import use
import matplotlib
from FLAlgorithms.users.userSP import UserSP
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data, get_gen, get_logit, get_gen_with_optim_latent
from FLAlgorithms.trainmodel.generator import G_EMNIST, G_mnist, Generator
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn as nn


class FedSP(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        self.args = args

        # Initialize data for all  users
        data = read_data(args.dataset)
        total_users = len(data[0])
        self.use_adam = 'adam' in self.algorithm.lower()
        print("Users in total: {}".format(total_users))


        self.kd_weight = []
        for i in range(total_users):
            id, train_data , test_data = read_user_data(i, data, dataset=args.dataset)
            user = UserSP(args, id, model, train_data, test_data, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
            self.kd_weight.append(user.count_class)
    

        self.kd_weight = torch.Tensor(np.array(self.kd_weight))
        self.kd_weight = self.kd_weight/torch.sum(self.kd_weight, axis=0, keepdim=True)
        self.kd_weight = self.kd_weight.cuda()

    
        print("Number of users / total users:",args.num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

        self.generator = G_EMNIST().to(args.device)
        if ".ckpt" in self.args.pretrained_G:
            self.generator.load_state_dict(torch.load(self.args.pretrained_G))
        self.generator.eval()


    def train(self, args, writer):

        if self.args.train_with_img:
            print("Training with image scheme")
        else:
            print("Training with latent scheme")

        if self.args.track_KL:
            self.track_KL = {}
            for i in range(len(self.users)):
                self.track_KL[i] = []

        best_acc = 0
        opt = torch.optim.SGD(
            self.model.parameters(), momentum=0.9, lr=args.KD_lr, weight_decay=5e-4
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.num_glob_iters)

        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users = self.select_users(glob_iter, self.num_users)

            self.send_parameters(mode='all')
            

            # Evaluate server
            glob_acc = self.evaluate(writer=writer, glob_iter=glob_iter)
            if glob_acc > best_acc:
                best_acc = glob_acc

            # Select user
            self.selected_users, user_idxs = self.select_users(glob_iter, self.num_users)

            # pseudo and latent
            all_latents = []
            all_pseudos = []
            all_images = []
            
            # Train users with their data 
            for i, user in enumerate(self.selected_users): # allow selected users to train
                user.train(glob_iter) 
                # user.train_latent_mapping(self.generator, glob_iter)  
                
                if not self.args.train_with_img:
                    imgs, latents = get_gen(self.generator, self.args.KD_latent_size, int(self.args.KD_num_per_client/self.args.KD_selective_feature_ratio))
                    imgs, latents = imgs.detach(), latents.detach()
                    logs, latents = get_logit(imgs, user.model, latents)
                    logs = F.softmax(logs / self.args.KD_temperature, dim=1)

                    if self.args.KD_selective_feature_ratio == 1.0:
                        selected_latents, selected_logs = latents.detach(), logs.detach()
                    else:
                        selected_latents, selected_logs = self.entropy_select_logits(glob_iter, logs, latents, user.count_class, frac=self.args.KD_selective_feature_ratio)

                    all_latents.append(selected_latents.detach())
                    all_pseudos.append(selected_logs.detach())
                else:
                    imgs, logs = self.get_data_from_client(user, self.args.batch_size, self.args.KD_num_per_client, self.args.KD_temperature)

                    all_images.append(imgs.detach())
                    all_pseudos.append(logs.detach())


            if not self.args.train_with_img:               
                all_latents = torch.stack(all_latents)
                all_pseudos = torch.stack(all_pseudos)
                if self.args.track_KL:
                    self.train_server_with_latent_track_KL(all_latents, all_pseudos, self.args, opt, user_idxs, self.track_KL)
                else:
                    self.train_server_with_latent(all_latents, all_pseudos, self.args, opt)
            else:
                all_pseudos = torch.stack(all_pseudos)
                all_images = torch.stack(all_images)

                if self.args.track_KL:
                    self.train_server_with_img_track_KL(all_images, all_pseudos, self.args, opt, user_idxs, self.track_KL)
                else:
                    self.train_server_with_img(all_images, all_pseudos, self.args, opt)

            if self.args.track_KL:
                if glob_iter == 200:
                    self.save_KL(glob_iter)
                if glob_iter == 499:
                    self.save_KL(glob_iter)                    

            scheduler.step()
                
        
        self.save_model()

    def save_KL(self, global_round):
        with open("runs/{}/KL_track_{}.txt".format(self.args.writer, global_round), 'a') as f:
            for client in self.track_KL:
                info_i = "{} : ".format(client)
                for KL_d in self.track_KL[client]:
                    info_i += "{}, ".format(KL_d)
                f.write(info_i)
                f.write("\n")

            f.close()

    def get_data_from_client(self, user, local_batchsize, num_img, KD_temperature):
        imgs = []
        logs = []

        while len(imgs) < int(num_img/local_batchsize):
            result_in = user.get_next_train_batch(count_labels=False)
            im_in = result_in['X'].cuda()
            lo_in = user.model(im_in)['output_z']
            lo_in = F.softmax(lo_in / KD_temperature, dim=1)
            
            if im_in.shape[0] == local_batchsize and lo_in.shape[0] == local_batchsize:
                imgs.append(im_in)
                logs.append(lo_in)

        imgs = torch.stack(imgs)
        imgs = self.magic_combine(imgs, 0, 2)
        logs = torch.stack(logs)
        logs = self.magic_combine(logs, 0, 2)

        return imgs, logs

    def check_entropy(self, logits, count_class):
        logs = logits.cpu()

        from torch.distributions import Categorical
        rank_entropy = []
        for log_idx in range(logs.shape[0]):
            log = logs[log_idx, :]

            E = False
            if E:
                log = torch.mul(log, count_class)
            
            entropy = Categorical(probs = log).entropy()
            rank_entropy.append(entropy.item())

        rank_entropy = torch.Tensor(rank_entropy)
        print("Mean entropy: {}".format(torch.mean(rank_entropy)))
        print("STD entropy: {}".format(torch.std(rank_entropy)))
        print("\n")

    def entropy_select_logits(self, glob_iter, logs, latent, count_class, frac=0.5):
        count_class_onehot = [1 if i > 0 else 0 for i in count_class]
        count_class_onehot = torch.Tensor(count_class_onehot)
        logs = logs.cpu()
        latent = latent.cpu()


        from torch.distributions import Categorical
        rank_entropy = []
        for log_idx in range(logs.shape[0]):
            log = logs[log_idx, :]

            E = False
            if E:
                log = torch.mul(log, count_class_onehot)
            
            entropy = Categorical(probs = log).entropy()
            rank_entropy.append(entropy.item())

        # rank_entropy = np.array(rank_entropy)
        # rank_entropy = np.argpartition(rank_entropy, -int(logs.shape[0]*frac))[]

        rank_entropy = torch.Tensor(rank_entropy)
        _, topk_index = torch.topk(rank_entropy, int(logs.shape[0]*frac), largest=False)

        selected_logs = torch.index_select(logs, 0, topk_index)
        selected_latents = torch.index_select(latent, 0, topk_index)

        

        # if glob_iter == 9:
        #     # print("Min Entro"torch.min(rank_entropy))
        #     # print(torch.max(rank_entropy))
        #     print("Mean entropy: {}".format(torch.mean(rank_entropy)))
        #     print("STD entropy: {}".format(torch.std(rank_entropy)))
        #     print("\n")

        #     # for i in topk_index:
        #         # print(logs[i])
        #         # exit()
        #     # for log_idx in range(selected_logs.shape[0]):
        #     #     log = logs[log_idx, :] 
        #     #     print(count_class)
        #     #     print(log)
        #     #     print(Categorical(probs = log).entropy())
        #     #     print("\n")

        #     #     if log_idx == 100:
        #     #         break
            
    
        return selected_latents.cuda(), selected_logs.cuda()

    def ce_select_logits(self, logs, latent, frac=0.5):
        logs = logs.cpu()
        latent = latent.cpu()

        from torch.distributions import Categorical
        rank_entropy = []
        for log_idx in range(logs.shape[0]):
            log = logs[log_idx, :]
            entropy = Categorical(probs = F.softmax(log)).entropy()
            rank_entropy.append(entropy.item())
        
        rank_entropy = torch.Tensor(rank_entropy)
        _, topk_index = torch.topk(rank_entropy, int(logs.shape[0]*frac), largest=False)

        selected_logs = torch.index_select(logs, 0, topk_index)
        selected_latents = torch.index_select(latent, 0, topk_index)

        return selected_latents.cuda(), selected_logs.cuda()

    def compute_KL_S_C(self, user_index, server):
        # Get data from client
        imgs, logs_C = self.get_data_from_client(self.users[user_index], self.args.batch_size, self.args.KD_num_per_client, self.args.KD_temperature)
        
        logs_S, _ = get_logit(imgs, server, None)

        KL_loss = self.kdloss(logs_S, logs_C, weight=1)
        
        return KL_loss

    def train_server_with_latent(self, latents, pseudo, args, opt):
        self.generator.eval()
        self.model.train()

        pseudo = self.magic_combine(pseudo, 0, 2)
        latents = self.magic_combine(latents, 0, 2)
        
        img = self.generator(latents).detach()

        data = self.Dataset_Server(img, pseudo)
        dataloader = torch.utils.data.DataLoader(data, batch_size=args.KD_bs, shuffle=True)

        # logit server
        total_loss = 0
        for i in range(args.KD_epoch):
            loss_i = 0
            count = 0
            for im, p in dataloader:
                opt.zero_grad()
                logit, _ = get_logit(im, self.model, None)
                loss = self.kdloss(logit, p, weight=self.args.KD_weight, temperature=self.args.KD_temperature)
                loss.backward()
                loss_i += loss
                count += 1
                opt.step() 
            # loss_i.backward()
            loss_i /= count
            total_loss += loss_i
        total_loss /= args.KD_epoch
        print("Server KD loss: {}".format(total_loss))

        self.model.eval()

    def train_server_with_latent_track_KL(self, latents, pseudo, args, opt, users_training_idx, track_KL):
        self.generator.eval()
        self.model.train()

        pseudo = self.magic_combine(pseudo, 0, 2)
        latents = self.magic_combine(latents, 0, 2)
        
        img = self.generator(latents).detach()

        data = self.Dataset_Server(img, pseudo)
        dataloader = torch.utils.data.DataLoader(data, batch_size=args.KD_bs, shuffle=True)

        # logit server
        total_loss = 0
        for i in range(args.KD_epoch):
            loss_i = 0
            count = 0
            for im, p in dataloader:
                opt.zero_grad()
                logit, _ = get_logit(im, self.model, None)
                loss = self.kdloss(logit, p, weight=self.args.KD_weight, temperature=self.args.KD_temperature)

                loss.backward()
                loss_i += loss
                count += 1
                opt.step() 
            # loss_i.backward()
            loss_i /= count
            total_loss += loss_i
        total_loss /= args.KD_epoch
        print("Server KD loss: {}".format(total_loss))

        for user_index in users_training_idx:
            d = self.compute_KL_S_C(user_index, self.model)

            track_KL[user_index].append(d.item())

        self.model.eval()
    
    def train_server_with_img_track_KL(self, imgs, pseudo, args, opt, users_training_idx, track_KL):
        self.generator.eval()
        self.model.train()

        imgs = self.magic_combine(imgs, 0, 2)
        pseudo = self.magic_combine(pseudo, 0, 2)


        data = self.Dataset_Server(imgs, pseudo)
        dataloader = torch.utils.data.DataLoader(data, batch_size=args.KD_bs, shuffle=True)

        # logit server
        total_loss = 0
        for i in range(args.KD_epoch):
            loss_i = 0
            count = 0
            for im, p in dataloader:
                im, p = im.cuda(), p.cuda() 

                opt.zero_grad()
                logit, _ = get_logit(im, self.model, None)
                loss = self.kdloss(logit, p, weight=1)
                loss.backward()
                loss_i += loss.item()
                count += 1
                opt.step() 
            loss_i /= count
            total_loss += loss_i

        total_loss /= args.KD_epoch
        print("Server KD loss: {}".format(total_loss))

        for user_index in users_training_idx:
            d = self.compute_KL_S_C(user_index, self.model)

            track_KL[user_index].append(d.item())

        self.model.eval()

    def train_server_with_img(self, imgs, pseudo, args, opt):
        self.generator.eval()
        self.model.train()

        imgs = self.magic_combine(imgs, 0, 2)
        pseudo = self.magic_combine(pseudo, 0, 2)


        data = self.Dataset_Server(imgs, pseudo)
        dataloader = torch.utils.data.DataLoader(data, batch_size=args.KD_bs, shuffle=True)

        # logit server
        total_loss = 0
        for i in range(args.KD_epoch):
            loss_i = 0
            count = 0
            for im, p in dataloader:
                im, p = im.cuda(), p.cuda() 

                opt.zero_grad()
                logit, _ = get_logit(im, self.model, None)
                loss = self.kdloss(logit, p, weight=1)
                loss.backward()
                loss_i += loss.item()
                count += 1
                opt.step() 
            loss_i /= count
            total_loss += loss_i

        total_loss /= args.KD_epoch
        print("Server KD loss: {}".format(total_loss))

        self.model.eval()

    def magic_combine(self, x, dim_begin, dim_end):
        combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
        return x.view(combined_shape)

    def kdloss(self, y, teacher_scores, temperature=3, weight=40):
        """
        Loss used for previous KD experiments
        """
        p = F.log_softmax(y / temperature, dim=1)
        # q = F.softmax(teacher_scores / temperature, dim=1)
        # l_kl = F.kl_div(p, q, reduction='batchmean')
        l_kl = F.kl_div(p, teacher_scores, reduction='batchmean')
        return l_kl * weight   

    class Dataset_Server(torch.utils.data.Dataset):
        'Characterizes a dataset for PyTorch'
        def __init__(self, X, y):
            'Initialization'
            self.y = y
            self.X = X

        def __len__(self):
            'Denotes the total number of samples'
            return len(self.y)

        def __getitem__(self, index):
            'Generates one sample of data'
            y = self.y[index]
            X = self.X[index]

            return X, y

    def visulize_img(self, tensor, output_name="temp"):
        import matplotlib.pyplot as plt
        img = tensor[0].permute(1, 2, 0).cpu().detach().numpy()
        plt.imshow(img)
        plt.savefig("{}.png".format(output_name))