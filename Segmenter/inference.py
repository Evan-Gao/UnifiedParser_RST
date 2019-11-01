import sys
import os
import json
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from tqdm import tqdm
import pickle
import torch
import numpy as np
import random
from torch.backends import cudnn
import argparse
from model import PointerNetworks
from solver import InferenceSolver

import os


def parse_args():
    parser = argparse.ArgumentParser(description='Pointer')

    parser.add_argument('-hdim', type=int, default=64, help='hidden size')
    parser.add_argument('-rnn', type=str, default='GRU', help='rnn type')
    parser.add_argument('-rnnlayers', type=int, default=6, help='how many rnn layers')
    parser.add_argument('-fine', type=str,default='False', help='fine tuning word embedding')
    parser.add_argument('-isbi', type=str, default='True', help='is bidirctional')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-dout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('-wd', type=float, default=0.0001, help='weight decay')
    parser.add_argument('-seed', type=int, default=550, help='random seed')
    parser.add_argument('-bsize', type=int, default=80, help='batch size')
    parser.add_argument('-lrdepoch', type=int, default=10, help='lr decay each epoch')
    parser.add_argument('-isbarnor', type=str, default='True', help='batch normalization')
    parser.add_argument('-iscudnn', type=str, default='True', help='cudnn')
    parser.add_argument('-modelpath', type=str, default=r'segmodel.torchsave', help='pretrained model path')
    args = parser.parse_args()

    return args


def dump_pickle(data, savepath):
    with open(savepath, 'wb') as fh:
        print('Saving {}'.format(savepath), end="   ...   ")
        pickle.dump(data, fh)
        print('Done!')


def load_pickle(loadpath):
    with open(loadpath, 'rb') as fh:
        print('Loading {}'.format(loadpath), end="   ...   ")
        dataset = pickle.load(fh)
        print('Done!')
    return dataset


if __name__ == '__main__':

    args = parse_args()
    use_cuda = True

    if args.iscudnn =='False':
        iscudnn = False
    else:
        iscudnn = True

    cudnn.enabled = iscudnn
    h_Dim = args.hdim
    rnn_type = args.rnn
    rnn_layers = args.rnnlayers
    lr = args.lr
    dout = args.dout
    wd = args.wd
    myseed = args.seed
    batch_size = args.bsize
    lrdepoch = args.lrdepoch

    torch.manual_seed(myseed)
    if use_cuda:
        torch.cuda.manual_seed_all(myseed)
    np.random.seed(myseed)
    random.seed(myseed)



    if args.isbi =='False':
        IS_BI = False
    else:
        IS_BI = True


    if args.fine =='False':
        FineTurning = False
    else:
        FineTurning = True


    if args.isbarnor == 'False':
        isbarnor = False
    else:
        isbarnor = True

    print(FineTurning)
    print(type(FineTurning))
    print(type(IS_BI))

    # edu_sample = load_pickle('Testing_EDUBreaks_seg.pickle')
    # sent_sample = load_pickle('Testing_InputSentences_seg.pickle')

    # tr_x = ['Indiana \'s Temporary Assistance for Needy Families ( TANF ) program provides cash assistance to families with dependent children and requires adults to work or prepare for work .'.split(' '),
    #         'Recipients will be provided with supportive assistance while on TANF including : child care services , transportation services , and vehicle repairs .'.split(' ')]

    # tr_x = sent_sample[:3]

    # loadpath = r'/home/lin/Segmentation/ELMo/SegData'
    # tr_x = pickle.load(open(os.path.join(loadpath,"Training_InputSentences_seg.pickle"),"rb"))
    # tr_y = pickle.load(open(os.path.join(loadpath,"Training_EDUBreaks_seg.pickle"), "rb"))
    #
    # dev_x = pickle.load(open(os.path.join(loadpath,"Testing_InputSentences_seg.pickle"),"rb"))
    # dev_y = pickle.load(open(os.path.join(loadpath,"Testing_EDUBreaks_seg.pickle"), "rb"))


    # filename = 'elmoLarge_dot_'+str(myseed) + 'seed_' + str(h_Dim) +'hidden_'+ \
    #            str(IS_BI)+'bi_' + rnn_type + 'rnn_' + str(FineTurning)+'Fined_'+str(rnn_layers)+\
    #            'rnnlayers_'+ str(lr)+'lr_'+str(dout)+'dropout_'+str(wd)+'weightdecay_'+str(batch_size)+'bsize_'+str(lrdepoch)+'lrdepoch_'+\
    #             str(isbarnor)+'barnor_'+str(iscudnn)+'iscudnn'
    #
    # save_path = os.path.join(args.savepath,filename)
    # print(save_path)


    word_dim=1024
    hidden_dim=h_Dim
    is_bi_encoder_rnn= IS_BI
    rnn_type=rnn_type
    rnn_layers=rnn_layers
    dropout_prob=dout
    use_cuda=use_cuda
    finedtuning=FineTurning
    isbanor=isbarnor


    model = PointerNetworks(word_dim=1024, hidden_dim=h_Dim,is_bi_encoder_rnn=IS_BI,
                            rnn_type=rnn_type,rnn_layers=rnn_layers,
                            dropout_prob=dout,use_cuda=use_cuda,finedtuning=FineTurning, isbanor=isbarnor)

    # load pretrained weights
    checkpoint = torch.load(args.modelpath, map_location=lambda storage, loc: storage)
    model_state_dict = checkpoint.state_dict()
    model.load_state_dict(model_state_dict, strict=False)

    if use_cuda:
        model = model.cuda()

    mysolver = InferenceSolver(model,
                               batch_size=batch_size,
                               eval_size=600,
                               epoch=1000,
                               lr=lr,
                               lr_decay_epoch=lrdepoch,
                               weight_decay=wd,
                               use_cuda=use_cuda)

    # load sharc snippet
    with open('/export/home/sharc/e3-pycharm/sharc/snippet_{}.json'.format('dev')) as f:
        fsnippet_dev = json.load(f)
    edus = {}
    for k, v in tqdm(fsnippet_dev.items()):
        boundary_start, boundary_end = [], []
        for clause in v['t_clauses']:
            clause_tokenized = list(token['sub'].strip() for token in clause)
            if len(clause_tokenized) == 1:
                boundary_start.append([0])
                boundary_end.append([0])
            else:
                clause_boundary_end, clause_boundary_start = mysolver.inference([clause_tokenized])
                boundary_start.append(clause_boundary_start[0].tolist()[0])
                boundary_end.append(clause_boundary_end[0].tolist()[0])
        edus[k] = [boundary_start, boundary_end]

    # save all edus
    with open('snippet_dev_edu.json', 'wt') as f:
        json.dump(edus, f, indent=2)
    # load ftree
    with open('../../sharc/e3/sharc/trees_dev.json') as f:
        ftree_dev = json.load(f)
    with open('../../sharc/e3/sharc/json/sharc_dev.json') as f:
        fraw_dev = json.load(f)
    # build dict for tree key to initial question & scenario
    fscini_dev = {}
    for ex in fraw_dev:
        if ex['tree_id'] not in fscini_dev.keys():
            fscini_dev[ex['tree_id']] = {
                'scenario': ex['scenario'],
                'initial': ex['question'],
            }
    # save boundary information into files, also compare with extracted rules
    with open('yfgao_snippet_edus_dev.txt', 'w', encoding='utf-8') as f:
        for tree_key in sorted(ftree_dev.keys()):
            tree = ftree_dev[tree_key]
            snippet = ftree_dev[tree_key]['snippet']
            f.write(tree_key + '\n')
            f.write('SNIPPET: \n{}\n'.format(snippet))
            f.write('SCENARIO: {}\n'.format(fscini_dev[tree_key]['scenario']))
            f.write('INITIAL: {}\n'.format(fscini_dev[tree_key]['initial']))
            f.write('-' * 5 + 'Discourse Segmentation' + '-' * 5 + '\n')
            boundary_start, boundary_end = edus[tree_key]
            for idx, clause in enumerate(fsnippet_dev[tree_key]['t_clauses']):
                clause_boundary_start, clause_boundary_end = boundary_start[idx], boundary_end[idx]
                clause_sub = list(token['sub'] for token in clause)
                for bs, be in zip(clause_boundary_start, clause_boundary_end):
                    f.write(' '.join(clause_sub[bs:be + 1]) + '\n')
            f.write('-' * 5 + 'Minimal Edit Distance' + '-' * 5 + '\n')
            for ques, span in tree['match_text'].items():
                f.write('QUES: {} \n'.format(ques))
                f.write('SPAN: {} \n'.format(span))
            f.write('\n\n')





    print('debug')


    # with open(os.path.join(args.savepath,'resultTable.csv'), 'a') as f:
    #   f.write(filename + ',' + ','.join(map(str,[best_i, best_pre, best_rec, best_f1]))+ '\n')
