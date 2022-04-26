from typing_extensions import Required
import torch
from FLAlgorithms.users.userbase import User
from torch.autograd import Variable
import torch.nn.functional as F


class UserSP(User):
    def __init__(self,  args, id, model, train_data, test_data, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)

        self.count_class = self.count_instance_per_class()

        self.args = args

    def count_instance_per_class(self):
        count = []
        for i in range(self.unique_labels):
            count.append(0)

        a = 0
        for datapoint in self.train_data:
            a+=1
            _, y = datapoint
            count[y.item()] += 1

        return count

    def train(self, glob_iter, lr_decay=True):
        self.model.cuda()
        self.model.train()

        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for i in range(self.K):
                result = self.get_next_train_batch()
                X, y = result['X'].cuda(), result['y'].cuda()
                self.optimizer.zero_grad()
                output=self.model(X)['output']

                loss=self.loss(output, y)
                loss.backward()
                self.optimizer.step()#self.plot_Celeb)

            # local-model <=== self.model
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
        if lr_decay:
            self.lr_scheduler.step(glob_iter)

        

    def train_latent_mapping(self, generator, glob_iter, lr_decay=True):
        generator.eval()
        self.model.eval()
        self.latent_mapping.train()

        for epoch in range(1, self.local_epochs + 1):
            for i in range(self.K):
                result = self.get_next_train_batch()
                X, y = result['X'].cuda(), result['y'].cuda()
                if y.shape[0] > 1:
                    z = torch.randn(y.shape[0], self.args.KD_latent_size, 1, 1).cuda()
                    z = Variable(z, requires_grad=True)

                    self.optimizer_latent_mapping.zero_grad()
                    z_op = self.latent_mapping(z)
                    z_op = z_op.reshape(y.shape[0], self.args.KD_latent_size, 1, 1)
                    X_hat = generator(z_op)
                    y_hat = self.model(X_hat)['output']
           
                    y_real = self.model(X)['output'].detach()
                    
                    loss = self.ce_loss(y_hat, y_real)
                    loss.backward()

                    self.optimizer_latent_mapping.step()

        if lr_decay:
            self.lr_scheduler_latent_mapping.step(glob_iter)



