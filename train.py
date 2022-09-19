
import os

import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy
from transformers import AdamW
import  copy
from sklearn import metrics
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data_utils import   Process_Corpus_BERT

from tqdm import tqdm
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
import json
from transformers import  AutoTokenizer
from MyModel import  ADIBERTCustom


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert_name)
      
        self.opt.lebel_dim = len(json.load(open('datasets/{0}/labels.json'.format(opt.dataset))))
        self.trainset = Process_Corpus_BERT(opt.dataset_file['train'], tokenizer, opt.max_seq_len, opt.dataset)
        self.valset = Process_Corpus_BERT(opt.dataset_file['dev'], tokenizer, opt.max_seq_len, opt.dataset)
        self.testset = Process_Corpus_BERT(opt.dataset_file['test'], tokenizer, opt.max_seq_len, opt.dataset)

        print(len( self.trainset), len( self.testset), len( self.valset))
        self.model = ADIBERTCustom(opt)
        self.model.to(opt.device)
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
       


    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader,t_total, labels):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None

        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            targets_all, outputs_all = None, None
            # switch model to training mode
            loss_total=[]
            self.model.train()
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                # self.model.train()
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]

                outputs= self.model(inputs)
                # targets = sample_batched['label'].to(self.opt.device)
                targets= inputs[-1]
                # targets= inputs[-1]
                loss = criterion(outputs, targets)

                # logger.info(outputs.shape)
                loss.sum().backward()

                optimizer.step()
                with torch.no_grad():
                    n_total += len(outputs)
                    loss_total.append(loss.sum().detach().item())




            logger.info('epoch : {}'.format(epoch))
            logger.info('loss: {:.4f}'.format(np.mean(loss_total)))
            pres, recall, f1_score, acc,cls_report = self._evaluate_acc_f1(val_data_loader, labels=labels)
            logger.info(cls_report)
            logger.info('> val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f},  val_acc: {:.4f}'.format(pres, recall, f1_score, acc))
            if f1_score > max_val_acc:
                max_val_acc = f1_score
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')

                path = copy.deepcopy(self.model.state_dict())

            lr_this_step = self.opt.learning_rate * self.warmup_linear(global_step / t_total,
                                                                       self.opt.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step


            # self.model.train()
        return path

    def _evaluate_acc_f1(self, data_loader, labels):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(tqdm(data_loader)):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_inputs[-1]
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().detach().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets.detach()
                    t_outputs_all = t_outputs.detach()
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets.detach()), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs.detach()), dim=0)
            true= t_targets_all.cpu().detach().numpy().tolist()
            pred =torch.argmax(t_outputs_all, -1).cpu().detach().numpy().tolist()
            f= metrics.f1_score(true,pred,average='macro')
            r= metrics.recall_score(true,pred,average='macro')
            p= metrics.precision_score(true,pred,average='macro')
            classification_repo= metrics.classification_report(true, pred, target_names=list(labels.keys()))
            acc= metrics.accuracy_score(true,pred)

        return  p, r, f, acc, classification_repo


    def run(self):
        # Loss and Optimizer
        if self.opt.dataset =='Corpus-2':
            labels={'MSA':0, 'Dialect':1}
        else:
            labels = json.load(open('datasets/{0}/labels.json'.format(self.opt.dataset)))
        
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(self.model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)


        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        t_total= int(len(train_data_loader) * self.opt.num_epoch)



        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader, t_total, labels=labels)
        self.model.load_state_dict(best_model_path)
        self.model.eval()
        pres, recall, f1_score, acc, cls_report= self._evaluate_acc_f1(test_data_loader, labels=labels)
        logger.info(cls_report)
        logger.info(
            '>> test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}, test_acc: {:.4f}'.format(pres, recall, f1_score, acc))
        with open('results_BERT.txt', 'a+') as f:
            f.write('{} {} >> test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}, test_acc: {:.4f}\n'.format(self.opt.baseline, self.opt.dataset,pres, recall, f1_score, acc))
        f.close()
        



def main(dataset=None, baseline=None, device_group=None):
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Corpuss-2', type=str, help='Corpus-9,Corpus-6, Nadi, QADI ')
    parser.add_argument('--baseline', default='camel', type=str, help='camel, mdb, arbert')
    parser.add_argument('--learning_rate', default=3e-5, type=float,)
    parser.add_argument('--adam_epsilon', default=2e-8, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--reg', type=float, default=0.00005, help='regularization constant for weight penalty')
    parser.add_argument('--num_epoch', default=5, type=int )
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--batch_size_val', default=128, type=int)
    parser.add_argument('--log_step', default=35500, type=int)
    parser.add_argument('--embed_dim', default=768, type=int)
    parser.add_argument('--hidden_dim', default=768, type=int)
    parser.add_argument('--max_grad_norm', default=10, type=int)
    parser.add_argument('--warmup_proportion', default=0.01, type=float)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--max_seq_len', default=60, type=int)
    parser.add_argument('--lebel_dim', default=3, type=int)
    parser.add_argument('--device', default='cuda' , type=str, help='e.g. cuda:0')
    parser.add_argument('--device_group', default='1' , type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=65, type=int, help='set seed for reproducibility')
    opt = parser.parse_args()



    if dataset is not  None:
        opt.dataset = dataset
    if baseline is not  None:
        opt.baseline = baseline
    if device_group is not  None:
        opt.device_group = device_group

    plm = {
        'camel':'CAMeL-Lab/bert-base-arabic-camelbert-msa',
        'arbert':'aubmindlab/bert-base-arabertv2',
        'mdb':'bashar-talafha/multi-dialect-bert-base-arabic'
    }
    opt.pretrained_bert_name= plm.get(opt.baseline)


    opt.seed= random.randint(20,300)

    if opt.seed is not None:

        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset_files = {
        'train': 'datasets/{0}/train.json'.format(opt.dataset),
        'test': 'datasets/{0}/test.json'.format(opt.dataset),
        'dev': 'datasets/{0}/dev.json'.format(opt.dataset)
    }
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_group
    input_colses =  ['input_ids', 'segments_ids', 'input_mask', 'label']
    
    opt.dataset_file = dataset_files
    opt.inputs_cols = input_colses
    opt.initializer = torch.nn.init.xavier_uniform_
    opt.optimizer = AdamW
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}-{}.log'.format(opt.dataset, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    logger.info('seed {}'.format(opt.seed))
    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
   

