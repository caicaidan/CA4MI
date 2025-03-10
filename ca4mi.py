import os
import sys
import time
from copy import deepcopy
import torch.nn as nn
import utils
import numpy as np
import torch
from scipy.linalg import fractional_matrix_power
import random
sys.path.append('../')


class CA4MI(object):

    def __init__(self, model, args, network):
        self.align = args.align
        self.mixup = args.mixup
        self.orth = args.orth
        self.use_prototypes = args.use_prototypes
        self.n_samples = args.n_samples
        self.alpha=args.alpha
        self.device = args.device
        self.checkpoint = args.checkpoint

        self.network = network
        self.inputsize = args.inputsize
        self.nepochs = args.n_epochs
        self.sbatch = args.batch_size
        self.latent_dim = args.latent_dim
        self.experiment = args.experiment

        self.encoder_lr = [args.encoder_lr] * args.n_subjects
        self.discriminator_lr = [args.discriminator_lr] * args.n_subjects
        self.lr_min = args.lr_min
        self.lr_factor = args.lr_factor
        self.lr_patience = args.lr_patience

        self.model = model
        self.args = args
        self.discriminator = self.initialize_discriminator(0)  # Initialize discriminator for subject 0
        self.encoder_optimizer = self.setup_encoder_optimizer(0)
        self.discriminator_optimizer = self.setup_discriminator_optimizer(0)

        self.cls_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.adversarial_loss_d = torch.nn.CrossEntropyLoss().to(self.device)
        self.adversarial_loss_s = torch.nn.CrossEntropyLoss().to(self.device)
        self.ort_loss = OrthLoss().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.adv_loss_reg = args.adv
        self.ort_loss_reg = args.orth_reg
        self.pro_loss_reg = args.pro_loss_reg
        self.encoder_step = args.shared_step
        self.discriminator_step = args.discriminator_step

        self.mu = 0.0
        self.sigma = 1.0
        self.subject_memory = {sub_id: {'x': [], 'y': [], 'sub_module_label': [], 'dis_label': []} for sub_id in range(args.n_subjects)}

    def get_prototype_samples(self, loader, net, sub_id):
        net.eval()
        all_fea = []
        all_output = []
        all_label = []

        with torch.no_grad():
            for batch, (data, target, sub_module_label, dis_label) in enumerate(loader):
                data = data[:, np.newaxis, :, :]
                x = data.to(self.device)
                y = target.to(self.device, dtype=torch.long)
                sub_module_label = sub_module_label.to(self.device)

                outputs = net(x, x, sub_module_label, sub_id)
                shared_out, private_out = net.get_encoded(x, x, sub_id)

                all_fea.append(shared_out)
                all_output.append(outputs)
                all_label.append(y)

        all_fea = torch.cat(all_fea).to(self.device)
        all_output = torch.cat(all_output).to(self.device)
        all_label = torch.cat(all_label).to(self.device)

        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(predict == all_label).item() / float(all_label.size(0))

        class_prototypes = []
        for i in range(all_output.size(1)):
            idx = (all_label == i).nonzero(as_tuple=True)[0]
            class_fea = all_fea[idx]
            class_proto = class_fea.mean(dim=0)
            class_prototypes.append(class_proto)

        class_prototypes = torch.stack(class_prototypes).to(self.device)
        print('Memory updated by adding {} prototype n_samples'.format(len(class_prototypes)))

        return class_prototypes, accuracy

    def reservoir_sampling(self, prototypes_list, max_prototypes):
        """
        Use reservoir sampling to maintain a fixed number of prototypes

        Parameters:
        prototypes_list -- List of prototypes, each element is a prototype tensor for a class
        max_prototypes -- Maximum number of prototypes to keep

        Returns:
        List of prototypes after sampling
        """
        all_prototypes = torch.cat(prototypes_list, dim=0)

        if all_prototypes.size(0) <= max_prototypes:
            return [all_prototypes]

        reservoir = all_prototypes[:max_prototypes].clone()

        for i in range(max_prototypes, all_prototypes.size(0)):
            j = random.randint(0, i)
            if j < max_prototypes:
                reservoir[j] = all_prototypes[i]

        return [reservoir]

    def compute_prototype_loss(self, features, targets, prototypes):
        pro_loss = torch.tensor(0).to(self.device, dtype=torch.float32)
        if prototypes is not None:
            for i in range(len(targets)):
                target = targets[i].item()
                if target < len(prototypes):
                    pro_loss += torch.norm(features[i] - prototypes[target])
            pro_loss /= len(targets)
        return pro_loss

    def IEA(self,x):
        cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            cov[i] = np.cov(x[i])
        refEA = np.mean(cov, 0)
        sqrtRefEA = fractional_matrix_power(refEA, -0.5)
        XEA = np.zeros(x.shape)
        for i in range(x.shape[0]):
            XEA[i] = np.dot(sqrtRefEA, x[i])
        return XEA

    def mixup_data(self, x, y, alpha, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def Adversarial_training(self, sub_id, dataset, prototypes=None):
        self.discriminator = self.initialize_discriminator(sub_id)

        best_loss = np.inf
        best_model = deepcopy(self.model.state_dict())
        best_loss_d = np.inf
        best_model_d = deepcopy(self.discriminator.state_dict())

        dis_lr_update = True
        discriminator_lr = self.discriminator_lr[sub_id]
        patience_d = self.lr_patience
        self.discriminator_optimizer = self.setup_discriminator_optimizer(sub_id, discriminator_lr)

        encoder_lr = self.encoder_lr[sub_id]
        patience = self.lr_patience
        self.encoder_optimizer = self.setup_encoder_optimizer(sub_id, encoder_lr)

        for e in range(self.nepochs):
            clock0 = time.time()

            # Merged train_epoch functionality here
            self.model.train()
            self.discriminator.train()

            if prototypes is not None and len(prototypes) > 0:
                if isinstance(prototypes, list) and all(isinstance(proto, torch.Tensor) for proto in prototypes):
                    prototypes = [proto.mean(dim=0) for proto in prototypes if proto.nelement() > 0]
                else:
                    raise ValueError("prototypes must be a list of non-empty Tensors")

            for data, target, sub_module_label, dis_label in dataset['train']:
                if self.align == 'yes':
                    aligned_data = self.IEA(data.numpy())
                    data = torch.tensor(aligned_data[:, np.newaxis, :, :], dtype=torch.float32, device=self.device)
                elif self.align == 'no':
                    data = data[:, np.newaxis, :, :].to(device=self.device, dtype=torch.float32)
                target = target.to(device=self.device, dtype=torch.long)
                sub_module_label = sub_module_label.to(device=self.device)

                x, y = data, target
                t_real_D, t_fake_D = dis_label.to(self.device), torch.zeros_like(dis_label).to(self.device)

                if self.mixup == 'yes':
                    alpha = self.alpha
                    data, targets_a, targets_b, lam = self.mixup_data(x, target, alpha, use_cuda=True)
                    x_sub_module = self.assign_subject_specific_mask(data, sub_module_label, sub_id)

                    for step in range(self.encoder_step):
                        self.encoder_optimizer.zero_grad()
                        self.model.zero_grad()
                        output = self.model(x, x_sub_module, sub_module_label, sub_id)
                        cls_loss = self.mixup_criterion(self.criterion, output, targets_a, targets_b, lam)
                        shared_encoded, private_encoded = self.model.get_encoded(x, x_sub_module, sub_id)
                        adv_loss = self.adversarial_loss_s(self.discriminator(shared_encoded, t_real_D, sub_id),
                                                           t_real_D)
                        ort_loss = self.ort_loss(shared_encoded, private_encoded) if self.orth == 'yes' else torch.tensor(0).to(
                            self.device)

                        pro_loss = self.compute_prototype_loss(private_encoded, y, prototypes) if prototypes else torch.tensor(0).to(
                            self.device)
                        total_loss = cls_loss + self.adv_loss_reg * adv_loss + self.ort_loss_reg * ort_loss + (
                            self.pro_loss_reg * pro_loss if self.use_prototypes == 'yes' else 0)
                        total_loss.backward(retain_graph=True)
                        self.encoder_optimizer.step()

                elif self.mixup == 'no':
                    x_sub_module = self.assign_subject_specific_mask(x, sub_module_label, sub_id)

                    for step in range(self.encoder_step):
                        self.encoder_optimizer.zero_grad()
                        self.model.zero_grad()
                        output = self.model(x, x_sub_module, sub_module_label, sub_id)
                        cls_loss = self.criterion(output, y)
                        shared_encoded, private_encoded = self.model.get_encoded(x, x_sub_module, sub_id)
                        adv_loss = self.adversarial_loss_s(self.discriminator(shared_encoded, t_real_D, sub_id),
                                                           t_real_D)
                        ort_loss = self.ort_loss(shared_encoded,
                                                 private_encoded) if self.orth == 'yes' else torch.tensor(0).to(
                            self.device)

                        pro_loss = self.compute_prototype_loss(private_encoded, y, prototypes) if prototypes else torch.tensor(0).to(
                            self.device)

                        total_loss = cls_loss + self.adv_loss_reg * adv_loss + self.ort_loss_reg * ort_loss + (
                            self.pro_loss_reg * pro_loss if self.use_prototypes == 'yes' else 0)
                        total_loss.backward(retain_graph=True)
                        self.encoder_optimizer.step()

                for step in range(self.discriminator_step):
                    self.discriminator_optimizer.zero_grad()
                    self.discriminator.zero_grad()
                    shared_encoded = self.model.get_encoded(x, x_sub_module, sub_id)[0].detach()
                    dis_real_loss = self.adversarial_loss_d(self.discriminator(shared_encoded, t_real_D, sub_id),
                                                            t_real_D)
                    dis_real_loss.backward(retain_graph=True)
                    z_fake = torch.randn((x.size(0), self.latent_dim), dtype=torch.float32,
                                         device=self.device) * self.sigma + self.mu
                    dis_fake_loss = self.adversarial_loss_d(self.discriminator(z_fake, t_real_D, sub_id), t_fake_D)
                    dis_fake_loss.backward(retain_graph=True)
                    self.discriminator_optimizer.step()

            clock1 = time.time()
            train_res = self.eval_(dataset['train'], sub_id, prototypes)
            utils.report_tr(train_res, e, self.sbatch, clock0, clock1)

            if (self.experiment in ['bci-competition-IV2a', 'bci-competition-IV2b', 'openBMI']) and e == 4:
                random_chance, threshold = 20., 22.
                if train_res['acc_t'] < threshold:
                    discriminator_lr, encoder_lr = discriminator_lr / 10., encoder_lr / 10.
                    self.discriminator_optimizer, self.encoder_optimizer = self.setup_discriminator_optimizer(sub_id, discriminator_lr), self.setup_encoder_optimizer(
                        sub_id, encoder_lr)
                    self.discriminator = self.initialize_discriminator(sub_id)
                    self.model = self.load_current_models(sub_id - 1) if sub_id > 0 else self.network.Cls_HEAD(
                        self.args).to(self.args.device)

            valid_res = self.eval_(dataset['valid'], sub_id)
            utils.report_val(valid_res)

            if valid_res['loss_tot'] < best_loss:
                best_loss, best_model, patience = valid_res['loss_tot'], deepcopy(
                    self.model.state_dict()), self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    encoder_lr /= self.lr_factor
                    if encoder_lr < self.lr_min:
                        print()
                        break
                    patience, self.encoder_optimizer = self.lr_patience, self.setup_encoder_optimizer(sub_id, encoder_lr)
                    print(f' lr={encoder_lr:.1e}', end='')

            if train_res['loss_a'] < best_loss_d:
                best_loss_d, best_model_d, patience_d = train_res['loss_a'], deepcopy(
                    self.discriminator.state_dict()), self.lr_patience
            else:
                patience_d -= 1
                if patience_d <= 0 and dis_lr_update:
                    discriminator_lr /= self.lr_factor
                    if discriminator_lr < self.lr_min:
                        dis_lr_update = False
                        print("Dis lr reached minimum value\n")
                    patience_d, self.discriminator_optimizer = self.lr_patience, self.setup_discriminator_optimizer(sub_id, discriminator_lr)
                    print(f' Dis lr={discriminator_lr:.1e}')

            print()

        self.model.load_state_dict(deepcopy(best_model))
        self.discriminator.load_state_dict(deepcopy(best_model_d))
        self.save_models(sub_id)

    def eval_(self, data_loader, sub_id, prototypes=None):

        loss_a, loss_t, loss_d, loss_total = 0, 0, 0, 0
        correct_d, correct_t = 0, 0
        num = 0
        batch = 0

        self.model.eval()
        self.discriminator.eval()

        res = {}
        with torch.no_grad():
            for batch, (data, target, sub_module_label, dis_label) in enumerate(data_loader):
                if self.align == 'yes':
                    aligned_data = self.IEA(data.numpy())
                    aligned_data = aligned_data[:, np.newaxis, :, :]
                    data = torch.tensor(aligned_data, dtype=torch.float32, device=self.device)
                elif self.align == 'no':
                    data = data[:, np.newaxis, :, :]
                    data = data.to(device=self.device, dtype=torch.float32)
                target = target.to(device=self.device, dtype=torch.long)
                sub_module_label = sub_module_label.to(device=self.device)

                x = data
                y = target
                t_real_D = dis_label.to(self.device)

                output = self.model(x, x, sub_module_label, sub_id)
                shared_out, private_out = self.model.get_encoded(x, x, sub_id)
                _, pred = output.max(1)
                correct_t += pred.eq(y.view_as(pred)).sum().item()

                output_d = self.discriminator.forward(shared_out, t_real_D, sub_id)
                _, pred_d = output_d.max(1)
                correct_d += pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

                cls_loss = self.cls_loss(output, y)
                adv_loss = self.adversarial_loss_d(output_d, t_real_D)

                if self.orth == 'yes':
                    ort_loss = self.ort_loss(shared_out, private_out)
                else:
                    ort_loss = torch.tensor(0).to(device=self.device, dtype=torch.float32)
                    self.ort_loss_reg = 0

                pro_loss = self.compute_prototype_loss(private_out, y, prototypes)

                if self.use_prototypes == 'yes':
                    total_loss = cls_loss + self.adv_loss_reg * adv_loss + self.ort_loss_reg * ort_loss + self.pro_loss_reg * pro_loss
                else:
                    total_loss = cls_loss + self.adv_loss_reg * adv_loss + self.ort_loss_reg * ort_loss

                loss_t += cls_loss
                loss_a += adv_loss
                loss_d += ort_loss
                loss_total += total_loss

                num += x.size(0)

        res['loss_t'], res['acc_t'] = loss_t.item() / (batch + 1), 100 * correct_t / num
        res['loss_a'], res['acc_d'] = loss_a.item() / (batch + 1), 100 * correct_d / num
        res['loss_d'] = loss_d.item() / (batch + 1)
        res['loss_tot'] = loss_total.item() / (batch + 1)
        res['size'] = utils.loader_size(data_loader=data_loader)

        return res

    def save_models(self, sub_id):
        print("Saving all models for subject {} ...".format(sub_id + 1))
        dis = deepcopy(self.discriminator.state_dict())
        torch.save({'model_state_dict': dis},
                   os.path.join(self.checkpoint, 'discriminator_{}.pth.tar'.format(sub_id)))

        model = deepcopy(self.model.state_dict())
        torch.save({'model_state_dict': model}, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(sub_id)))
    def load_current_models(self, sub_id):
        print("Loading checkpoint for subject {} ...".format(sub_id))
        net = self.network.Cls_HEAD(self.args)
        checkpoint = torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(sub_id)), weights_only=True)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.to(self.args.device)
        return net

    def load_previous_model(self, sub_id):
        # Load a previous model
        net = self.network.Cls_HEAD(self.args)
        checkpoint = torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(sub_id)), weights_only=True)
        net.load_state_dict(checkpoint['model_state_dict'])
        current_shared_module = deepcopy(self.model.shared.state_dict())
        net.shared.load_state_dict(current_shared_module)
        encoder = net.to(self.args.device)
        return encoder

    def test_previous_subjects(self, data_loader, sub_id, model, prototypes=None):
        loss_a, loss_t, loss_d, loss_total = 0, 0, 0, 0
        correct_d, correct_t = 0, 0
        num = 0
        batch = 0

        model.eval()
        self.discriminator.eval()
        res = {}

        with torch.no_grad():
            for batch, (data, target, sub_module_label, dis_label) in enumerate(data_loader):
                if self.align == 'yes':
                    aligned_data = self.IEA(data.numpy())
                    aligned_data = aligned_data[:, np.newaxis, :, :]
                    data = torch.tensor(aligned_data, dtype=torch.float32, device=self.device)
                elif self.align == 'no':
                    data = data[:, np.newaxis, :, :]
                    data = data.to(device=self.device, dtype=torch.float32)
                target = target.to(device=self.device, dtype=torch.long)
                sub_module_label = sub_module_label.to(device=self.device)

                x = data
                y = target
                t_real_D = dis_label.to(self.device)

                output = model.forward(x, x, sub_module_label, sub_id)
                shared_out, private_out = model.get_encoded(x, x, sub_id)

                _, pred = output.max(1)
                correct_t += pred.eq(y.view_as(pred)).sum().item()

                output_d = self.discriminator.forward(shared_out, sub_module_label, sub_id)
                _, pred_d = output_d.max(1)
                correct_d += pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

                if self.orth == 'yes':
                    ort_loss = self.ort_loss(shared_out, private_out)
                else:
                    ort_loss = torch.tensor(0).to(device=self.device, dtype=torch.float32)
                    self.ort_loss_reg = 0

                adv_loss = self.adversarial_loss_d(output_d, t_real_D)
                cls_loss = self.cls_loss(output, y)


                pro_loss = self.compute_prototype_loss(private_out, y, prototypes)

                if self.use_prototypes == 'yes':
                    total_loss = cls_loss + self.adv_loss_reg * adv_loss + self.ort_loss_reg * ort_loss + self.pro_loss_reg * pro_loss
                else:
                    total_loss = cls_loss + self.adv_loss_reg * adv_loss + self.ort_loss_reg * ort_loss

                loss_t += cls_loss
                loss_a += adv_loss
                loss_d += ort_loss
                loss_total += total_loss

                num += x.size(0)

        res['loss_t'], res['acc_t'] = loss_t.item() / (batch + 1), 100 * correct_t / num
        res['loss_a'], res['acc_d'] = loss_a.item() / (batch + 1), 100 * correct_d / num
        res['loss_d'] = loss_d.item() / (batch + 1)
        res['loss_tot'] = loss_total.item() / (batch + 1)
        res['size'] = utils.loader_size(data_loader=data_loader)

        return res

    def assign_subject_specific_mask(self, data, sub_module_label, sub_id):
        t_current = sub_id * torch.ones_like(sub_module_label)
        body_mask = torch.eq(t_current, sub_module_label).to(self.device)
        x_sub_module = data.clone()

        for index in range(data.size(0)):
            if not body_mask[index]:
                x_sub_module[index] = x_sub_module[index].detach()

        return x_sub_module.to(self.device)

    def setup_encoder_optimizer(self, sub_id, encoder_lr=None):
        if encoder_lr is None: encoder_lr = self.encoder_lr[sub_id]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=encoder_lr, weight_decay=self.args.encoder_wd)
        return optimizer

    def setup_discriminator_optimizer(self, sub_id, discriminator_lr=None):
        if discriminator_lr is None: discriminator_lr = self.discriminator_lr[sub_id]
        optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr, weight_decay=self.args.discriminator_wd)
        return optimizer

    def initialize_discriminator(self, sub_id):
        discriminator = Discriminator(self.args, sub_id).to(self.args.device)
        return discriminator


class Discriminator(torch.nn.Module):
    def __init__(self, args, sub_id):
        super(Discriminator, self).__init__()

        latent_dim = args.latent_dim
        hidden1 = args.hidden_dim[0]
        hidden2 = args.hidden_dim[1]

        if args.orth == 'yes':
            self.dis = torch.nn.Sequential(
                GradientReversal(args.lam),
                torch.nn.Linear(latent_dim, hidden1),
                torch.nn.ELU(),
                torch.nn.Linear(hidden1, hidden2),
                torch.nn.Linear(hidden2, sub_id + 2)
            )
        else:
            self.dis = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, hidden1),
                torch.nn.ELU(),
                torch.nn.Linear(hidden1, hidden2),
                torch.nn.Linear(hidden2, sub_id + 2)
            )

    def forward(self, z, labels, sub_id):
        return self.dis(z)


class GradientReversalFunction(torch.autograd.Function):
    """
    From:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/cb65581f20b71ff9883dd2435b2275a1fd4b90df/utils.py#L26

    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    """
    Gradient Reversal Layer as introduced in domain adversarial neural networks.
    Multiplies the gradient by `-lambda_` during the backward pass, allowing
    adversarial learning.

    Args:
        lambda_ (float): Scaling factor for the reversed gradient.

    Methods:
        forward(x): Applies the gradient reversal operation.

    Returns:
        Tensor: Input tensor `x` unchanged in the forward pass.
    """

    def __init__(self, lambda_):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)



class OrthLoss(torch.nn.Module):
    """
    Computes orthogonal loss to encourage feature separation between two
    feature matrices `D1` and `D2` by minimizing the dot product of their
    normalized representations, as described in "Domain Separation Networks"
    (https://arxiv.org/abs/1608.06019).

    Args:
        D1 (Tensor): First feature matrix of shape (batch_size, feature_dim).
        D2 (Tensor): Second feature matrix of shape (batch_size, feature_dim).

    Returns:
        Tensor: Scalar orthogonality loss.
    """

    def __init__(self):
        super(OrthLoss, self).__init__()

    def forward(self, D1, D2):
        D1 = D1.view(D1.size(0), -1)
        D1_norm = D1 / (torch.norm(D1, p=2, dim=1, keepdim=True).expand_as(D1) + 1e-6)

        D2 = D2.view(D2.size(0), -1)
        D2_norm = D2 / (torch.norm(D2, p=2, dim=1, keepdim=True).expand_as(D2) + 1e-6)

        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))











