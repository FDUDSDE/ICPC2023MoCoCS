import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
import copy
from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

from parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index,
                    tree_to_ast_token,
                    tree_to_ast_node,
                    nodes_to_code,
                    tree_leaf)

from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}
# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


def extract_ast(code, parser, lang):
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    if lang == "php":
        code = "<?php"+code+"?>"
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    ast_nodes = []
    ast_tokens = []
    tree_to_ast_node(root_node, ast_nodes)
    tree_to_ast_token(root_node, ast_tokens, bytes(code, 'utf8'))
    return ast_nodes, ast_tokens


def extract_dataflow(code, parser, lang):
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    if lang == "php":
        code = "<?php"+code+"?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 ast_tokens,
                 ast_ids,
                 dfg_tokens,
                 dfg_ids,
                 position_idx,
                 ast_to_code,
                 ast_leaf,
                 dfg_to_dfg,
                 url,
                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.ast_tokens = ast_tokens
        self.ast_ids = ast_ids
        self.dfg_tokens = dfg_tokens
        self.dfg_ids = dfg_ids
        self.position_idx = position_idx
        self.ast_to_code = ast_to_code
        self.ast_leaf = ast_leaf
        self.dfg_to_dfg = dfg_to_dfg
        self.url = url


def convert_examples_to_features(js, tokenizer, args):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    # code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    parser = parsers[args.lang]
    ast_nodes, ast_tokens = extract_ast(js['original_string'], parser, args.lang)
    ast_tokens = tokenizer.tokenize(' '.join(ast_tokens)[:args.ast_length-4])
    ast_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]+ast_tokens+[tokenizer.sep_token]
    # ast_tokens = [tokenizer.cls_token]+ast_tokens+[tokenizer.sep_token]
    ast_ids = tokenizer.convert_tokens_to_ids(ast_tokens)
    padding_length = args.ast_length - len(ast_ids)
    ast_ids += [tokenizer.pad_token_id]*padding_length
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(ast_tokens))]
    position_idx += [tokenizer.pad_token_id]*padding_length
    length = len([tokenizer.cls_token]) + len([tokenizer.sep_token]) + 1
    # length = len([tokenizer.cls_token])
    ast_to_code = nodes_to_code(ast_nodes)
    ast_leaf = tree_leaf(ast_nodes)
    ast_to_code = [(x[0]+length, x[1]+length) for x in ast_to_code]
    ast_leaf = [x+length for x in ast_leaf]

    _, dfg = extract_dataflow(js['original_string'], parser, args.lang)
    dfg_tokens = [x[0] for x in dfg]
    dfg_tokens = tokenizer.tokenize(' '.join(dfg_tokens)[:args.dfg_length-4])
    dfg_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]+dfg_tokens+[tokenizer.sep_token]
    # dfg_tokens = [tokenizer.cls_token]+dfg_tokens+[tokenizer.sep_token]
    dfg_ids = tokenizer.convert_tokens_to_ids(dfg_tokens)
    padding_length = args.dfg_length - len(dfg_ids)
    dfg_ids += [tokenizer.pad_token_id]*padding_length
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]

    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    # nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, ast_tokens, ast_ids, dfg_tokens, dfg_ids, position_idx, ast_to_code, ast_leaf, dfg_to_dfg, js['url'] if "url" in js else js["retrieval_idx"])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.args = args
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase" in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)

        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))

        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        attn_mask = np.zeros((self.args.ast_length, self.args.ast_length), dtype=bool)
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        for idx, i in enumerate(self.examples[item].ast_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        for idx, (a, b) in enumerate(self.examples[item].ast_to_code):
            if a < self.args.ast_length and b < self.args.ast_length:
                attn_mask[idx, a:b] = True
                attn_mask[a:b, idx] = True
            elif a < self.args.ast_length:
                attn_mask[idx, a:max_length] = True
                attn_mask[a:max_length, idx] = True
        for a in self.examples[item].ast_leaf:
            for b in self.examples[item].ast_leaf:
                if a < self.args.ast_length and b < self.args.ast_length:
                    attn_mask[a, b] = True

        attn_mask_dfg = np.zeros((self.args.dfg_length, self.args.dfg_length), dtype=bool)
        max_length_dfg = len(self.examples[item].dfg_tokens)
        for idx, i in enumerate(self.examples[item].dfg_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length_dfg] = True
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+3 < self.args.dfg_length and idx+3 < self.args.dfg_length:
                    attn_mask_dfg[idx+3, a+3] = True

        return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(self.examples[item].nl_ids),
                torch.tensor(self.examples[item].ast_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(self.examples[item].dfg_ids),
                torch.tensor(attn_mask_dfg),
                )


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    """ Train the model """
    # get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    # get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*args.num_train_epochs)


    # train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)

    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    model.train()
    tr_num, tr_loss, best_mrr = 0, 0, 0
    for idx in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # get inputs
            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)
            ast_inputs = batch[2].to(args.device)
            attn_mask = batch[3].to(args.device)
            position_idx = batch[4].to(args.device)
            dfg_inputs = batch[5].to(args.device)
            attn_mask_dfg = batch[6].to(args.device)
            # get code and nl vectors
            loss, code_vec, nl_vec = model(code_inputs=code_inputs,nl_inputs=nl_inputs,ast_inputs=ast_inputs,dfg_inputs=dfg_inputs)

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1) % 100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step+1, round(tr_loss/tr_num, 5)))
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # evaluate
        results = evaluate(args, model, tokenizer,args.eval_data_file, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))

        # save best model
        if results['eval_mrr'] > best_mrr:
            best_mrr = results['eval_mrr']
            logger.info("  "+"*"*20)
            logger.info("  Best mrr:%s",round(best_mrr, 4))
            logger.info("  "+"*"*20)

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, file_name, eval_when_training=False):
    query_dataset = TextDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)

    code_dataset = TextDataset(tokenizer, args, args.codebase_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    code_vecs = []
    nl_vecs = []
    for batch in query_dataloader:
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)
            nl_vecs.append(nl_vec.cpu().numpy())

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())
    model.train()
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)

    scores = np.matmul(nl_vecs,code_vecs.T)

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]

    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)

    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    recall1 = []
    recall5 = []
    recall10 = []
    for url, sort_id in zip(nl_urls,sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1/rank)
            if rank <= 1:
                recall1.append(1)
            else:
                recall1.append(0)
            if rank <= 5:
                recall5.append(1)
            else:
                recall5.append(0)
            if rank <= 10:
                recall10.append(1)
            else:
                recall10.append(0)
        else:
            ranks.append(0)
            recall1.append(0)
            recall5.append(0)
            recall10.append(0)

    result = {
        "eval_mrr": float(np.mean(ranks)),
        "R@1": float(np.mean(recall1)),
        "R@5": float(np.mean(recall5)),
        "R@10": float(np.mean(recall10)),
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--ast_length", default=256, type=int,
                        help="Optional ast sep input sequence length after tokenization.")
    parser.add_argument("--dfg_length", default=128, type=int,
                        help="Optional dfg sep input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # set seed
    set_seed(args.seed)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path)
    model2 = copy.deepcopy(model)

    model = Model(model, model2, args)
    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

        # training
    if args.do_train:
        train(args, model, tokenizer)

    # evaluation
    results = {}
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer,args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 3)))

    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.test_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))


if __name__ == "__main__":
    main()
