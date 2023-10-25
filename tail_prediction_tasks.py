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
import pandas as pd
from tqdm import tqdm
from typing import Tuple

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
load = True
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

def create_constraints(triples: np.ndarray) -> Tuple[dict, dict, dict, dict]:
    """
    (1) Extract domains and ranges of relations
    (2) Store a mapping from relations to entities that are outside of the domain and range.
    Create constraints entities based on the range of relations
    :param triples:
    :return:
    """
    assert isinstance(triples, np.ndarray)
    assert triples.shape[1] == 3

    # (1) Compute the range and domain of each relation
    domain_per_rel = dict()
    range_per_rel = dict()

    range_constraints_per_rel = dict()
    domain_constraints_per_rel = dict()
    set_of_entities = set()
    set_of_relations = set()
    print(f'Constructing domain and range information by iterating over {len(triples)} triples...', end='\t')
    for (e1, p, e2) in triples:
        # e1, p, e2 have numpy.int16 or else types.
        domain_per_rel.setdefault(p, set()).add(e1)
        range_per_rel.setdefault(p, set()).add(e2)
        set_of_entities.add(e1)
        set_of_relations.add(p)
        set_of_entities.add(e2)
    print(f'Creating constraints based on {len(set_of_relations)} relations and {len(set_of_entities)} entities...',
          end='\t')
    for rel in set_of_relations:
        range_constraints_per_rel[rel] = list(set_of_entities - range_per_rel[rel])
        domain_constraints_per_rel[rel] = list(set_of_entities - domain_per_rel[rel])
    return domain_constraints_per_rel, range_constraints_per_rel, domain_per_rel, range_per_rel

def main():
    apply_semantic_constraints = False
    confidence_score = 0.0
    filtered_properties = False
    if Config.process: preprocess(Config.dataset, delete_data=False)
    input_keys = ['e1', 'rel', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(Config.dataset, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']

    num_entities = vocab['e1'].num_token

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

    if Config.cuda:
        model.cuda()

    if load:
        valid_tails_dict = pd.read_csv(f'valid_tails_dict.txt', sep='\t', header=None, usecols=[0, 1],names=['e1rel', 'e2_multi1_lower'],dtype=str, skiprows=[0])
        valid_tails = valid_tails_dict.set_index('e1rel')['e2_multi1_lower'].to_dict()
        df_valid_tails = pd.read_csv(f'valid_tails.txt', sep='\t', header=None, usecols=[0, 1, 2, 3, 4, 5, 6],names=['e1', 'rel', 'e2_multi1', 'e1_lower', 'rel_lower', 'e2_multi1_lower', 'e1rel'],dtype=str, skiprows=[0])

        model_params = torch.load(model_path)
        total_param_size = []
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            total_param_size.append(count)
        model.load_state_dict(model_params)
        model.eval()
        print("entities number", vocab["e1"].num_token)
        print("relations number", vocab["rel"].num_token)
        ranks = []
        rel_list = []
        if filtered_properties==False:
            for rel in vocab["rel"].token2idx:
                rel_list.append(rel)
        e1_list = []
        for item in vocab["e1"].idx2token:
            e1_list.append(False)
        df = pd.read_csv(f'../../KG-Abstractive-Summarization/data/ESBM/elist.txt', sep='\t', header=None, dtype=str, skiprows=[0])
        entities = list(set(df[2].values))
        for entity in tqdm(entities):
            entity_name = entity.split("/")[-1]
            if filtered_properties==True:
                df_rels = pd.read_csv(f'../../KG-Abstractive-Summarization/data/ESBM/relevant-properties/{entity_name}', sep='\t', header=None, dtype=str)
                rel_list = list(set(df_rels[0].values))
                rel_list.append("http://purl.org/dc/terms/subject")
            entity = entity.lower()
            try:
                check_token = [vocab["e1"].token2idx[f"{entity}"]]
            except:
                continue
            triples = dict()
            for rel in rel_list:
                rel = rel.lower()
                if rel=="oov" or rel=="":
                    continue
                try:
                    check_token = [vocab["rel"].token2idx[f"{rel}"]]
                except:
                    continue
                elist = e1_list.copy()
                head = entity
                relation = rel
                valid_tail_entities = valid_tails.get(f"({head},{relation})")
                if valid_tail_entities is None:
                    continue
                valid_tail_entities = valid_tail_entities.replace("[","")
                valid_tail_entities = valid_tail_entities.replace("]","")
                valid_tail_entities = valid_tail_entities.split("\',")
                valid_tail_embeddings = []
                for valid_tail_entity in valid_tail_entities:
                    valid_tail_entity = valid_tail_entity.replace("\'","")
                    if "\"," in valid_tail_entity:
                        valid_tail_entities = valid_tail_entity.split("\",")
                        for valid_tail_entity in valid_tail_entities:
                            valid_tail_entity = valid_tail_entity.replace("\"","")
                            try:
                                elist[vocab["e1"].token2idx[valid_tail_entity.strip()]]=True
                            except:
                                pass
                    else:
                        valid_tail_entity = valid_tail_entity.replace("\"","")
                        try:
                            elist[vocab["e1"].token2idx[valid_tail_entity.strip()]]=True
                        except:
                            pass
                rel_id = vocab["rel"].token2idx[rel]
                e1_tensor = torch.tensor([vocab["e1"].token2idx[f"{entity}"]]).cuda()
                rel_tensor = torch.tensor([vocab["rel"].token2idx[rel]]).cuda()
                scores = model.forward(e1_tensor, rel_tensor)
                if apply_semantic_constraints==True:
                    elist = torch.tensor(elist)
                    # Inverting the mask
                    not_allowed_entities_mask = ~elist
                    not_allowed_entities_mask = not_allowed_entities_mask[None, :]
                    # Applying constraints by setting scores of not allowed entities to -inf
                    scores[not_allowed_entities_mask] = -torch.inf
                pred_output = scores.view(1, -1).cpu()
                (output_top_scores, output_top) = torch.topk(pred_output, 5)
                scores = output_top_scores.squeeze(0).detach().numpy().tolist()
                pred_topk = output_top.squeeze(0).detach().numpy().tolist()
                predicted_tail = vocab["e1"].idx2token[pred_topk[0]]
                for num, score in enumerate(scores):
                    if score > confidence_score:
                        predicted_tail = vocab["e1"].idx2token[pred_topk[num]]
                        triple = (head, relation, predicted_tail)
                        triples[triple]=score
            if len(triples)>0:
                triples_sorted = sorted(triples.items(), key=lambda x:x[1], reverse=True)
                topk=len(triples_sorted)
                #if len(triples_sorted)>15:
                #    topk=15
                fw = open(f"results/{entity_name}.txt", "w")
                for triple_score in triples_sorted[:topk]:
                    triple, score = triple_score
                    h, r, t = triple
                    fw.write(f"{h}\t{r}\t{t}\n")
                fw.close()
if __name__ == '__main__':
    main()
