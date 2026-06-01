from dataloaders.loader_urfunny_v1 import URFUNNY_Loader
from dataloaders.loader_mosi import MOSI_Loader
from dataloaders.loader_mosei import MOSEI_Loader
from models.model import *
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_device, BTLoss, DJSLoss, HSICLoss



activations_dict = {
    "elu": nn.ELU, "relu": nn.ReLU, "gelu": nn.GELU, "prelu": nn.PReLU, "rrelu": nn.RReLU, "leakyrelu": nn.LeakyReLU,              
    "tanh": nn.Tanh, "hardtanh": nn.Hardtanh,
}

class Trainer(object):
    def __init__(self, configs, logger, save_dir, loader=None, writer=None):
        self.configs = configs
        self.logger = logger
        self.save_dir = save_dir
        self.loader = loader
        self.writer = writer
        
    def get_dataloaders(self):
        if self.loader is not None:
            loader = self.loader
        elif self.configs.dataset == 'URFUNNY':
            loader = URFUNNY_Loader(self.configs)
        elif self.configs.dataset == 'MOSI':
            loader = MOSI_Loader(self.configs)
        elif self.configs.dataset == 'MOSEI':
            loader = MOSEI_Loader(self.configs)

        self.train_dataloader = loader.train_dataloader
        self.valid_dataloader = loader.valid_dataloader
        self.test_dataloader = loader.test_dataloader
    
    def get_model(self):
        self.model = MyModel(self.configs)
    
    def get_optimizers(self):
        configs = self.configs
        lr = configs.lr
        if configs.optimizer == 'Adam':
            b1 = configs.adam_b1
            b2 = configs.adam_b2
            weight_decay = configs.adam_weight_decay
            self.optim = optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        elif configs.optimizer == 'AdamW':
            b1 = configs.adam_b1
            b2 = configs.adam_b2
            weight_decay = configs.adam_weight_decay
            self.optim = optim.AdamW(self.model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)

        if configs.scheduler == 'explr':
            gamma = configs.explr_gamma
            self.sched = optim.lr_scheduler.ExponentialLR(self.optim, gamma=gamma)
        elif configs.scheduler == 'plateau':
            patience = configs.plateau_patience
            factor = configs.plateau_factor
            self.sched = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', patience=patience, factor=factor, verbose=True)

    def build(self):
        self.configs.activation = activations_dict[self.configs.activation]

        self.get_dataloaders()
        self.get_model()
        self.get_optimizers()
        
        # To device
        if self.configs.use_mgpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.configs.gpu_ids
            self.gpu_ids = list(map(int, self.configs.gpu_ids.split(",")))
            self.device = get_device(self.configs.device)
        else:
            self.device = get_device(self.configs.device)
        
        if self.configs.use_mgpu:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
        
        self.model.to(self.device)

    def step_schedulers(self):
        if self.configs.scheduler == 'plateau':
            self.sched.step(self.valid_avg_pred_loss)
        else:
            self.sched.step()

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    def cal_metrics(self, y_true, y_pred, log_info=False):
        """
        Metric scheme adapted from:
        https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
        """
        if self.configs.dataset == 'URFUNNY':
            test_preds = np.argmax(y_pred, 1)
            test_truth = y_true

            if log_info:
                self.logger.info(f"Accuracy (pos/neg): {accuracy_score(test_truth, test_preds)}")
            
            return accuracy_score(test_truth, test_preds)

        elif self.configs.dataset in ['MOSI', 'MOSEI']:
            test_preds = y_pred
            test_truth = y_true

            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)

            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            
            corr = np.corrcoef(test_preds, test_truth)[0][1]
            f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
            mae = np.mean(np.absolute(test_preds - test_truth))   

            # non-neg - neg
            binary_truth_nn = (test_truth >= 0)
            binary_preds_nn = (test_preds >= 0)

            # pos - neg
            binary_truth = (test_truth[non_zeros] > 0)
            binary_preds = (test_preds[non_zeros] > 0)

            if log_info:
                self.logger.info(f"Accuracy (non-neg/neg): {accuracy_score(binary_truth_nn, binary_preds_nn)}")
                self.logger.info(f"Accuracy (pos/neg): {accuracy_score(binary_truth, binary_preds)}")
                self.logger.info(f"Mult_Acc: {mult_a7}")
                self.logger.info(f"Corr: {corr}")
                self.logger.info(f"F1: {f_score}")
                self.logger.info(f"MAE: {mae}")  
            
            return accuracy_score(binary_truth, binary_preds)

    def train(self):
        # To Find Best Model on Valid Set
        self.valid_best_record = {'epoch':0, 'valid_acc':0., 'valid_avg_pred_loss':float('inf')}
        self.num_trials = self.configs.rollback_num
        self.curr_patience = self.configs.rollback_patience
        # To Find Best Model on Test Set
        self.test_best_record = {'epoch':0, 'test_acc':0., 'test_avg_pred_loss':float('inf')}
        # ==================

        ext_lossfn = DJSLoss().to(self.device)
        bt_lossfn = BTLoss(self.configs).to(self.device)
        hsic_lossfn = HSICLoss().to(self.device)
        cyr_lossfn = nn.MSELoss().to(self.device)
        rec_lossfn = nn.MSELoss().to(self.device)
        
        if self.configs.dataset == 'URFUNNY':
            pred_lossfn = nn.CrossEntropyLoss(reduction="mean")
        elif self.configs.dataset == 'MOSI':
            pred_lossfn = nn.L1Loss(reduction="mean")
        elif self.configs.dataset == 'MOSEI':
            pred_lossfn = nn.MSELoss(reduction="mean")
            pred_lossfn2 = nn.L1Loss(reduction="mean")
        
        batch_num = len(self.train_dataloader)
        batch_size = self.configs.batch_size

        for epoch in range(self.configs.epoch_num):
            self.model.train()

            self.epoch = epoch
            self.logger.info(f'Training epoch {epoch}...')

            avg_info_loss_share = 0.0
            avg_info_loss_private  = 0.0
            avg_info_loss_noise = 0.0
            avg_info_loss = 0.0
            avg_bt_loss = 0.0
            avg_hsic_loss = 0.0
            avg_cyr_loss = 0.0
            avg_rec_loss = 0.0
            avg_noise_pred_loss = 0.0
            avg_pred_loss = 0.0
            avg_train_loss = 0.0
            
            train_acc = 0.0

            for step, batch in enumerate(self.train_dataloader):
                self.model.zero_grad()

                v, a, y, bert_sent, bert_sent_type, bert_sent_mask = batch

                # Set batch_first
                if self.configs.dataset in ['MOSEI', 'URFUNNY']:
                    v = v.permute([1,0,2])
                    a = a.permute([1,0,2])

                batch_V = v.to(self.device)
                batch_A = a.to(self.device)
                batch_T_bert = bert_sent.to(self.device)
                batch_T_bert_type = bert_sent_type.to(self.device)
                batch_T_bert_mask = bert_sent_mask.to(self.device)
                batch_Y = y.to(self.device)
                
                # Adaptive Noise_Pred ReverseLayer Setting
                p_of_alpha = float(step + epoch * batch_num) / self.configs.epoch_num / batch_num
                reverse_alpha = 2. / (1. + np.exp(-10 * p_of_alpha)) - 1
                # ====================================
                if self.configs.rev_alpha_type == 'alpha':
                    self.model.configs.reverse_alpha = reverse_alpha 
                elif self.configs.rev_alpha_type == '1':
                    self.model.configs.reverse_alpha = 1

                pred = self.model(batch_V, batch_A, batch_T_bert, batch_T_bert_type, batch_T_bert_mask)

                enc_v, enc_a, enc_t = self.model.enc_v, self.model.enc_a, self.model.enc_t
                
                outputs = self.model.outputs
                
                (cyr_feat_v, cyr_feat_a, cyr_feat_t, 
                cyr_noise_v, cyr_noise_a, cyr_noise_t) = (self.model.cyr_feat_v, self.model.cyr_feat_a, self.model.cyr_feat_t, 
                                                         self.model.cyr_noise_v, self.model.cyr_noise_a, self.model.cyr_noise_t)
                
                rec_v, rec_a, rec_t = self.model.rec_v, self.model.rec_a, self.model.rec_t
                
                noise_pred = self.model.noise_pred

                if self.configs.dataset == 'URFUNNY':
                    batch_Y = batch_Y.squeeze()
                elif self.configs.dataset in ['MOSI', 'MOSEI']:
                    pass
                
                batch_acc = self.cal_metrics(batch_Y.squeeze().detach().cpu().numpy(), 
                                             pred.squeeze().detach().cpu().numpy(), log_info=False)
                
                train_acc += batch_acc.item() * len(batch_Y) / (batch_num * batch_size)

                # Global Info Loss

                global_info_loss_share_v = ext_lossfn(T=outputs.global_mutual_share_v, T_prime=outputs.global_mutual_share_v_prime)
                global_info_loss_share_a = ext_lossfn(T=outputs.global_mutual_share_a, T_prime=outputs.global_mutual_share_a_prime)
                global_info_loss_share_t = ext_lossfn(T=outputs.global_mutual_share_t, T_prime=outputs.global_mutual_share_t_prime)
                    
                global_info_loss_private_v = ext_lossfn(T=outputs.global_mutual_private_v, T_prime=outputs.global_mutual_private_v_prime)
                global_info_loss_private_a = ext_lossfn(T=outputs.global_mutual_private_a, T_prime=outputs.global_mutual_private_a_prime)
                global_info_loss_private_t = ext_lossfn(T=outputs.global_mutual_private_t, T_prime=outputs.global_mutual_private_t_prime)

                global_info_loss_noise_v = ext_lossfn(T=outputs.global_mutual_noise_v, T_prime=outputs.global_mutual_noise_v_prime)
                global_info_loss_noise_a = ext_lossfn(T=outputs.global_mutual_noise_a, T_prime=outputs.global_mutual_noise_a_prime)
                global_info_loss_noise_t = ext_lossfn(T=outputs.global_mutual_noise_t, T_prime=outputs.global_mutual_noise_t_prime)
                
                info_loss_share = 1/3 * (global_info_loss_share_v + \
                                         global_info_loss_share_a + \
                                         global_info_loss_share_t)
            
                info_loss_private  = 1/3 * (global_info_loss_private_v  + \
                                         global_info_loss_private_a  + \
                                         global_info_loss_private_t)
                        
                info_loss_noise = 1/3 * (global_info_loss_noise_v + \
                                         global_info_loss_noise_a + \
                                         global_info_loss_noise_t)
                
                info_loss = 1. * info_loss_share + \
                            1. * info_loss_private + \
                            1. * info_loss_noise

                avg_info_loss_share += info_loss_share.item() / batch_num
                avg_info_loss_private += info_loss_private.item() / batch_num
                avg_info_loss_noise += info_loss_noise.item() / batch_num
                avg_info_loss += info_loss.item() / batch_num

                # Barlow Twins Loss
                bt_loss = 1/3 * (bt_lossfn(outputs.share_v, outputs.share_a) + \
                                 bt_lossfn(outputs.share_v, outputs.share_t) + \
                                 bt_lossfn(outputs.share_a, outputs.share_t))
                
                avg_bt_loss += bt_loss.item() / batch_num

                # HSIC Loss
                hsic_loss = 1/12 * (hsic_lossfn(outputs.share_v, outputs.private_v) + \
                                   hsic_lossfn(outputs.share_a, outputs.private_a) + \
                                   hsic_lossfn(outputs.share_t, outputs.private_t) + \
                                   hsic_lossfn(outputs.private_v, outputs.private_a) + \
                                   hsic_lossfn(outputs.private_v, outputs.private_t) + \
                                   hsic_lossfn(outputs.private_a, outputs.private_t)) + \
                            1/12 * (hsic_lossfn(outputs.share_v, outputs.noise_v) + \
                                   hsic_lossfn(outputs.share_a, outputs.noise_a) + \
                                   hsic_lossfn(outputs.share_t, outputs.noise_t) + \
                                   hsic_lossfn(outputs.private_v, outputs.noise_v) + \
                                   hsic_lossfn(outputs.private_a, outputs.noise_a) + \
                                   hsic_lossfn(outputs.private_t, outputs.noise_t))
                
                avg_hsic_loss += hsic_loss.item() / batch_num

                # Recon Loss
                rec_loss = 1/3 * (rec_lossfn(rec_v, enc_v) + rec_lossfn(rec_a, enc_a) + rec_lossfn(rec_t, enc_t))
                
                avg_rec_loss += rec_loss.item() / batch_num

                # DiCyR Loss, to minimize M.I., be careful to avoid info. to be pushed into single vector (DiCyR)
                
                cyr_loss = 1/6 * (cyr_lossfn(outputs.feat_v, cyr_feat_v) + \
                                  cyr_lossfn(outputs.feat_a, cyr_feat_a) + \
                                  cyr_lossfn(outputs.feat_t, cyr_feat_t) + \
                                  cyr_lossfn(outputs.noise_v, cyr_noise_v) + \
                                  cyr_lossfn(outputs.noise_a, cyr_noise_a) + \
                                  cyr_lossfn(outputs.noise_t, cyr_noise_t))
 
                avg_cyr_loss += cyr_loss.item() / batch_num

                # Noise_Pred Loss
                noise_pred_loss = pred_lossfn(noise_pred, batch_Y)

                avg_noise_pred_loss += noise_pred_loss.item() / batch_num

                # Pred Loss
                if self.configs.dataset == 'URFUNNY':
                    pred_loss = pred_lossfn(pred, batch_Y)
                elif self.configs.dataset == 'MOSI':
                    pred_loss = pred_lossfn(pred, batch_Y)
                elif self.configs.dataset == 'MOSEI':
                    pred_loss = pred_lossfn(pred, batch_Y) + pred_lossfn2(torch.clamp(pred, min=-3, max=3), torch.clamp(batch_Y, min=-3, max=3))
                
                avg_pred_loss +=  pred_loss.item() / batch_num

                # Train Loss
                train_loss = self.configs.info_loss_coeff * info_loss + \
                             self.configs.bt_loss_coeff * bt_loss + \
                             self.configs.hsic_loss_coeff * hsic_loss + \
                             self.configs.rec_loss_coeff * rec_loss + \
                             self.configs.cyr_loss_coeff * cyr_loss + \
                             self.configs.noise_pred_loss_coeff * noise_pred_loss + \
                             pred_loss
               
                avg_train_loss += train_loss.item() / batch_num

                train_loss.backward()             
                
                # Clip Grad
                if self.configs.clip_grad:
                    torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.configs.clip_ts)
                
                self.optim.step()
            
                if self.configs.use_tsboard:
                    self.writer.add_scalar('Loss(Train)', train_loss.item(), (epoch-1)*batch_num + step)
                
                if step % 100 == 0:
                    log_dict = {'pred_loss': pred_loss.item(),
                                'train_loss': train_loss.item(),
                                }

                    self.logger.info(f'[{epoch}/{self.configs.epoch_num-1}]-[{step}/{batch_num-1}]-Train, {log_dict}')
            
            self.logger.info(f'[{epoch}/{self.configs.epoch_num-1}]-Train, train_acc:{train_acc}, train_loss:{avg_train_loss}')
            
            self.logger.info(f'Validation epoch {epoch}...')
            break_flag = self.valid()
            
            self.logger.info(f'Testing epoch {epoch}...')
            self.test()

            if break_flag:
                self.logger.info("Running out of trials, early stopping.")
                break

        self.logger.info(f'Finished Training!')
        self.logger.info(f'Best validation record: {self.valid_best_record}')
        self.logger.info(f'Best test record: {self.test_best_record}')

    def valid(self):
        self.model.eval()

        epoch = self.epoch
        batch_num = len(self.valid_dataloader)

        y_true_list = []
        y_pred_list = []
        pred_loss_list = []

        if self.configs.dataset == 'URFUNNY':
            pred_lossfn = nn.CrossEntropyLoss(reduction="mean")
        elif self.configs.dataset == 'MOSI':
            pred_lossfn = nn.L1Loss(reduction="mean")
        elif self.configs.dataset == 'MOSEI':
            pred_lossfn = nn.MSELoss(reduction="mean")
            pred_lossfn2 = nn.L1Loss(reduction="mean")

        with torch.no_grad():
            for step, batch in enumerate(self.valid_dataloader):
                v, a, y, bert_sent, bert_sent_type, bert_sent_mask = batch

                # Set batch_first
                if self.configs.dataset in ['MOSEI', 'URFUNNY']:
                    v = v.permute([1,0,2])
                    a = a.permute([1,0,2])

                batch_V = v.to(self.device)
                batch_A = a.to(self.device)
                batch_T_bert = bert_sent.to(self.device)
                batch_T_bert_type = bert_sent_type.to(self.device)
                batch_T_bert_mask = bert_sent_mask.to(self.device)
                batch_Y = y.to(self.device)
                
                pred = self.model(batch_V, batch_A, batch_T_bert, batch_T_bert_type, batch_T_bert_mask)

                if self.configs.dataset == 'URFUNNY':
                    batch_Y = batch_Y.squeeze()
                
                if self.configs.dataset == 'URFUNNY':
                    pred_loss = pred_lossfn(pred, batch_Y)
                elif self.configs.dataset == 'MOSI':
                    pred_loss = pred_lossfn(pred, batch_Y)
                elif self.configs.dataset == 'MOSEI':
                    pred_loss = pred_lossfn(pred, batch_Y) + pred_lossfn2(torch.clamp(pred, min=-3, max=3), torch.clamp(batch_Y, min=-3, max=3))

                pred_loss_list.append(pred_loss.item())
                y_pred_list.append(pred.detach().cpu().numpy())
                y_true_list.append(batch_Y.detach().cpu().numpy())

                if self.configs.use_tsboard:
                    self.writer.add_scalar('Pred Loss(Valid)', pred_loss.item(), (epoch-1)*batch_num + step)
            
            avg_pred_loss = np.mean(pred_loss_list)
            y_pred = np.concatenate(y_pred_list, axis=0).squeeze()
            y_true = np.concatenate(y_true_list, axis=0).squeeze()

            valid_acc = self.cal_metrics(y_true, y_pred, log_info=False)

            self.logger.info(f'[{epoch}/{self.configs.epoch_num-1}]-Valid, valid_acc:{valid_acc}, avg_pred_loss:{avg_pred_loss}')
            self.valid_avg_pred_loss = avg_pred_loss

        if valid_acc >= self.valid_best_record['valid_acc']:
            self.valid_best_record['epoch'] = epoch
            self.valid_best_record['valid_acc'] = valid_acc
            self.valid_best_record['valid_avg_pred_loss'] = avg_pred_loss
            self.logger.info(f'Find best model on valid set, epoch:{epoch}, valid_acc:{valid_acc}, avg_pred_loss:{avg_pred_loss}')

            if self.configs.save_checkpoint:
                torch.save({'best_epoch':self.valid_best_record['epoch'],
                            'best_acc':self.valid_best_record['valid_acc'],
                            'best_avg_pred_loss':self.valid_best_record['valid_avg_pred_loss'],
                            'model_state_dict':self.model.state_dict(),
                            'optim_state_dict':self.optim.state_dict(),
                            },
                            os.path.join(self.save_dir, f'chkpt.pth.tar'))
            
            self.curr_patience = self.configs.rollback_patience
        else:
            self.curr_patience -= 1
            if self.curr_patience <= -1:
                if not self.configs.save_checkpoint:
                    return True
                
                self.logger.info('Running out of patience, loading previous best model on valid set.')
                self.num_trials -= 1
                self.curr_patience = self.configs.rollback_patience
                best_chkpt_root = os.path.join(self.save_dir, f'chkpt.pth.tar')
                checkpoint = torch.load(best_chkpt_root)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optim.load_state_dict(checkpoint['optim_state_dict'])
                self.step_schedulers()
                self.logger.info(f"Current learning rate: {self.optim.state_dict()['param_groups'][0]['lr']}")
            
        if self.num_trials < 0:
            return True
        
        return False
    
    def test(self):
        self.model.eval()

        epoch = self.epoch
        batch_num = len(self.test_dataloader)

        y_true_list = []
        y_pred_list = []
        pred_loss_list = []

        if self.configs.dataset == 'URFUNNY':
            pred_lossfn = nn.CrossEntropyLoss(reduction="mean")
        elif self.configs.dataset == 'MOSI':
            pred_lossfn = nn.L1Loss(reduction="mean")
        elif self.configs.dataset == 'MOSEI':
            pred_lossfn = nn.MSELoss(reduction="mean")
            pred_lossfn2 = nn.L1Loss(reduction="mean")

        with torch.no_grad():
            for step, batch in enumerate(self.test_dataloader):
                v, a, y, bert_sent, bert_sent_type, bert_sent_mask = batch

                # Set batch_first
                if self.configs.dataset in ['MOSEI', 'URFUNNY']:
                    v = v.permute([1,0,2])
                    a = a.permute([1,0,2])

                batch_V = v.to(self.device)
                batch_A = a.to(self.device)
                batch_T_bert = bert_sent.to(self.device)
                batch_T_bert_type = bert_sent_type.to(self.device)
                batch_T_bert_mask = bert_sent_mask.to(self.device)
                batch_Y = y.to(self.device)
                
                pred = self.model(batch_V, batch_A, batch_T_bert, batch_T_bert_type, batch_T_bert_mask)

                if self.configs.dataset == 'URFUNNY':
                    batch_Y = batch_Y.squeeze()
                
                if self.configs.dataset == 'URFUNNY':
                    pred_loss = pred_lossfn(pred, batch_Y)
                elif self.configs.dataset == 'MOSI':
                    pred_loss = pred_lossfn(pred, batch_Y)
                elif self.configs.dataset == 'MOSEI':
                    pred_loss = pred_lossfn(pred, batch_Y) + pred_lossfn2(torch.clamp(pred, min=-3, max=3), torch.clamp(batch_Y, min=-3, max=3))

                pred_loss_list.append(pred_loss.item())
                y_pred_list.append(pred.detach().cpu().numpy())
                y_true_list.append(batch_Y.detach().cpu().numpy())

                if self.configs.use_tsboard:
                    self.writer.add_scalar('Pred Loss(Test)', pred_loss.item(), (epoch-1)*batch_num + step)
            
            avg_pred_loss = np.mean(pred_loss_list)
            y_pred = np.concatenate(y_pred_list, axis=0).squeeze()
            y_true = np.concatenate(y_true_list, axis=0).squeeze()

            test_acc = self.cal_metrics(y_true, y_pred, log_info=True)

            self.logger.info(f'[{epoch}/{self.configs.epoch_num-1}]-Test, test_acc:{test_acc}, avg_pred_loss:{avg_pred_loss}')
        
        if test_acc >= self.test_best_record['test_acc']:
            self.test_best_record['epoch'] = epoch
            self.test_best_record['test_acc'] = test_acc
            self.test_best_record['test_avg_pred_loss'] = avg_pred_loss
            self.logger.info(f'Find best model on test set, epoch:{epoch}, test_acc:{test_acc}, avg_pred_loss:{avg_pred_loss}')
        
