import torch
import numpy as np
import random
import json
import os
from collections import defaultdict
from torch.backends import cudnn
from Segmenter.model import PointerNetworks
from Segmenter.solver import InferenceSolver
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

EDU_MIN_LENGTH = 30 # minimum length of clause for discourse segmentation
cudnn.enabled = True
myseed = 550
torch.manual_seed(myseed)
torch.cuda.manual_seed_all(myseed)
np.random.seed(myseed)
random.seed(myseed)

if __name__ == '__main__':
    for split in ['dev', 'train']:
        ftree = '../../sharc/sharc-kvmn/sharc/trees_{}.json'.format(split)
        with open(ftree) as f:
            mapping = json.load(f)
            # split long sentences into several edus
            stats = defaultdict(list)
            for ex in mapping.values():
                stats['clause_len'].extend([len(cla) for cla in ex['clauses_t']])
            print('clause_len:')
            v = stats['clause_len']
            print('mean: {}'.format(sum(v) / len(v)))
            print('median: {}'.format(sorted(v)[len(v)//2]))
            print('min: {}'.format(min(v)))
            print('max: {}'.format(max(v)))
            ftree_edu = '../../sharc/sharc-kvmn/sharc/trees_edu_{}.json'.format(split)

            model = PointerNetworks(word_dim=1024, hidden_dim=64, is_bi_encoder_rnn=True,
                                    rnn_type='GRU', rnn_layers=6, dropout_prob=0.2,
                                    use_cuda=True, finedtuning=False, isbanor=True)
            # load pretrained weights
            checkpoint = torch.load(r'segmodel.torchsave', map_location=lambda storage, loc: storage)
            model_state_dict = checkpoint.state_dict()
            model.load_state_dict(model_state_dict, strict=False)
            model = model.cuda()
            mysolver = InferenceSolver(model, batch_size=1, eval_size=600,
                                       epoch=1000, lr=0.01, lr_decay_epoch=10,
                                       weight_decay=0.0001, use_cuda=True)
            for ex in mapping.values():
                for clause in ex['clauses_t']:
                    if len(clause) >= EDU_MIN_LENGTH:
                        clause_boundary_end, clause_boundary_start = mysolver.inference([[tok['text'] for tok in clause]])