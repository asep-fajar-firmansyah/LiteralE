import json
import torch
import pickle
import numpy as np
import argparse
import sys
import os
import math

from os.path import join
import torch.backends.cudnn as cudnn

from evaluation import ranking_and_hits
from model import DistMultLiteral, ComplexLiteral, ConvELiteral, DistMultLiteral_gate,ComplexLiteral_gate, ConvELiteral_gate, DistMultLiteral_gate_text, ComplexLiteral_gate_text, ConvELiteral_gate_text

from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer
from spodernet.preprocessing.processors import ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, ApplyFunction, StreamToBatch
from spodernet.utils.global_config import Config, Backends
from spodernet.utils.logger import Logger, LogLevel
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
from spodernet.hooks import LossHook, ETAHook
from spodernet.utils.util import Timer
from spodernet.utils.cuda_utils import CUDATimer
from spodernet.utils.cuda_utils import CUDATimer
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
np.set_printoptions(precision=3)
import pdb
timer = CUDATimer()
cudnn.benchmark = True

# parse console parameters and set global variables
Config.backend = Backends.TORCH
Config.parse_argv(sys.argv)

Config.cuda = True
Config.embedding_dim = 200
#Logger.GLOBAL_LOG_LEVEL = LogLevel.DEBUG


# Random seed
from datetime import datetime
rseed = int(datetime.now().timestamp())
print(f'Random seed: {rseed}')
np.random.seed(rseed)
torch.manual_seed(rseed)
torch.cuda.manual_seed(rseed)


#model_name = 'DistMult_{0}_{1}'.format(Config.input_dropout, Config.dropout)
model_name = '{2}_{0}_{1}_literal'.format(Config.input_dropout, Config.dropout, Config.model_name)
epochs = Config.epochs
load = False
if Config.dataset is None:
    Config.dataset = 'FB15k-237'
model_path = 'saved_models/{0}_{1}.model'.format(Config.dataset, model_name)


''' Preprocess knowledge graph using spodernet. '''
def preprocess(dataset_name, delete_data=False):
    full_path = 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    train_path = 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    dev_ranking_path = 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    test_ranking_path = 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)

    keys2keys = {}
    keys2keys['e1'] = 'e1' # entities
    keys2keys['rel'] = 'rel' # relations
    #keys2keys['rel_eval'] = 'rel' # relations
    keys2keys['e2'] = 'e1' # entities
    keys2keys['e2_multi1'] = 'e1' # entity
    keys2keys['e2_multi2'] = 'e1' # entity
    input_keys = ['e1', 'rel', 'e2', 'e2_multi1', 'e2_multi2']
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))
    # process full vocabulary and save it to disk
    d.set_path(full_path)
    p = Pipeline(Config.dataset, delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    p.execute(d)
    p.save_vocabs()

    # process train, dev and test sets and save them to hdf5
    p.skip_transformation = False
    for path, name in zip([train_path, dev_ranking_path, test_ranking_path], ['train', 'dev_ranking', 'test_ranking']):
        d.set_path(path)
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
        p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), keys=['e1', 'rel', 'e2', 'e2_multi1', 'e2_multi2'])
        p.add_post_processor(StreamToHDF5(name, samples_per_file=1500 if Config.dataset == 'YAGO3-10' else 1000, keys=input_keys))
        p.execute(d)


def main():
    if Config.process: preprocess(Config.dataset, delete_data=True)
    input_keys = ['e1', 'rel', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(Config.dataset, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']

    if Config.epochs != 0:
        num_entities = vocab['e1'].num_token

        train_batcher = StreamBatcher(Config.dataset, 'train', Config.batch_size, randomize=True, keys=input_keys)
        dev_rank_batcher = StreamBatcher(Config.dataset, 'dev_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys)
        test_rank_batcher = StreamBatcher(Config.dataset, 'test_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys)

        # Load literals
        numerical_literals = np.load(f'data/{Config.dataset}/literals/numerical_literals.npy', allow_pickle=True)
        text_literals = np.load(f'data/{Config.dataset}/literals/text_literals.npy', allow_pickle=True)

        # Normalize numerical literals
        max_lit, min_lit = np.max(numerical_literals, axis=0), np.min(numerical_literals, axis=0)
        numerical_literals = (numerical_literals - min_lit) / (max_lit - min_lit + 1e-8)

        # Load literal models
        if Config.model_name is None or Config.model_name == 'DistMult':
            model = DistMultLiteral_gate(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals)
        elif Config.model_name == 'ComplEx':
            model = ComplexLiteral_gate(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals)
        elif Config.model_name == 'ConvE':
            model = ConvELiteral_gate(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals)
        elif Config.model_name == 'DistMult_text':
            model = DistMultLiteral_gate_text(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals, text_literals)
        elif Config.model_name == 'ComplEx_text':
            model = ComplexLiteral_gate_text(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals, text_literals)
        elif Config.model_name == 'ConvE_text':
            model = ConvELiteral_gate_text(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals, text_literals)
        elif Config.model_name == 'DistMult_glin':
            model = DistMultLiteral(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals)
        elif Config.model_name == 'ComplEx_glin':
            model = ComplexLiteral(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals)
        elif Config.model_name == 'ConvE_glin':
            model = ConvELiteral(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals)
        else:
            raise Exception("Unknown model!")

        train_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))

        eta = ETAHook('train', print_every_x_batches=100)
        train_batcher.subscribe_to_events(eta)
        train_batcher.subscribe_to_start_of_epoch_event(eta)
        train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=100))

        if Config.cuda:
            model.cuda()
        if load:
            model_params = torch.load(model_path)
            print(model)
            total_param_size = []
            params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
            for key, size, count in params:
                total_param_size.append(count)
                print(key, size, count)
            print(np.array(total_param_size).sum())
            model.load_state_dict(model_params)
            model.eval()
            ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')
            ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
        else:
            model.init()

        total_param_size = []
        params = [value.numel() for value in model.parameters()]
        print(params)
        print(np.sum(params))

        opt = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.L2)
        for epoch in range(epochs):
            model.train()
            for i, str2var in enumerate(train_batcher):
                opt.zero_grad()
                e1 = str2var['e1']
                rel = str2var['rel']
                e2_multi = str2var['e2_multi1_binary'].float()
                # label smoothing

                #e2_multi = ((1.0-Config.label_smoothing_epsilon)*e2_multi) + (1.0/e2_multi.size(1))
                pred = model.forward(e1, rel)
                loss = model.loss(pred, e2_multi)
                loss.backward()
                opt.step()

                train_batcher.state.loss = loss.cpu()


            print('saving to {0}'.format(model_path))
            torch.save(model.state_dict(), model_path)

            model.eval()
            with torch.no_grad():
                if epoch % 3 == 0:
                    if epoch > 0:
                        ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
                        ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')


if __name__ == '__main__':
    main()
