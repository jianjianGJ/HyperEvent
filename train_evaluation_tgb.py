import os
import csv
import torch
import time
import argparse
import logging
import numpy as np
from tqdm import tqdm
import yaml
from copy import deepcopy
import torch.nn as nn
from utils_data_tgb import (get_segment, load_ns_eval, 
                            get_batch, get_x_y)
from model import SequenceClassifier
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

def configure_logging(dataset):
    log_dir = f"logs/{dataset}"
    model_dir = f"models/{dataset}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(), model_dir

def record_results(dataset, params, loss, mrr, seed, avg_epoch_time, max_gpu_mem):
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, "results.csv")
    
    fieldnames = ['timestamp', 'dataset', 'mrr', 'loss', 
                  'num_neighbor', 'num_latest', 'big_hood',
                  'num_segment_train','num_segment_val','num_segment_test',
                  'd_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout',
                  'num_epochs', 'learning_rate',
                  'avg_epoch_time', 'max_gpu_mem_mb']  
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    row = {
        'timestamp': timestamp,
        'dataset': dataset,
        'mrr': mrr,
        'loss': loss,
        'num_neighbor': params['num_neighbor'],
        'num_latest': params['num_latest'],
        'big_hood': params['big_hood'],
        'num_segment_train': params['num_segment_train'],
        'num_segment_val': params['num_segment_val'],
        'num_segment_test': params['num_segment_test'],
        'd_model': params['d_model'],  # model
        'nhead': params['nhead'],
        'num_layers': params['num_layers'],
        'dim_feedforward': params['dim_feedforward'],
        'dropout': params['dropout'],
        'num_epochs': params['num_epochs'],
        'learning_rate': params['learning_rate'],
        'avg_epoch_time': avg_epoch_time,  
        'max_gpu_mem_mb': max_gpu_mem  
    }
    
    file_exists = os.path.isfile(result_file)
    with open(result_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        
def train_one_batch(model, x, y , criterion, opt):
    model.train()
    opt.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    opt.step()
    batch_loss = loss.item()
    return batch_loss

def evaluate(model, x, y, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        loss = criterion(outputs, y)
        batch_loss = loss.item()
    return batch_loss, outputs

def create_arg_parser():
    parser = argparse.ArgumentParser(description="Temporal Link Prediction")
    parser.add_argument("--dataset", required=True, 
                        choices=['tgbl-wiki', 'tgbl-review', 'tgbl-coin', 'tgbl-comment', 'tgbl-flight'])
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--num_neighbor", type=int, default=None)
    parser.add_argument("--num_latest", type=int, default=None)
    parser.add_argument("--big_hood", type=int, default=None)
    parser.add_argument("--num_segment_train", type=int, default=None)
    parser.add_argument("--num_segment_val", type=int, default=None)
    parser.add_argument("--num_segment_test", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dim_feedforward", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    return parser

def load_hyperparameters(experiment_name, config_path='config_tgb.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    if 'default' not in config:
        raise KeyError("Must include 'default' configuration")
    merged = deepcopy(config['default'])
    experiment_config = config.get(experiment_name, {}) or {}
    if not isinstance(experiment_config, dict):
        raise TypeError(f"Configuration for {experiment_name} must be a dictionary")
    merged.update(experiment_config)
    return merged

def merge_params(default_params, args):
    param_keys = ['num_runs', 'num_neighbor', 'num_latest','big_hood',
                  'num_segment_train','num_segment_val','num_segment_test',
                  'd_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout',
                 'num_epochs', 'learning_rate']
    merged = default_params.copy()
    for key in param_keys:
        if getattr(args, key) is not None:
            merged[key] = getattr(args, key)
    return merged

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    default_params = load_hyperparameters(args.dataset)
    params = merge_params(default_params, args)
    num_runs = params['num_runs']
    initial_seed = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger, model_dir = configure_logging(args.dataset)
    param_str_list = [f'{str(k)}:{str(v)}' for k,v in params.items()]
    logger.info("\n".join(param_str_list))
    
    dataset = PyGLinkPropPredDataset(root='datasets', name=args.dataset)
    min_dst_idx = int(dataset.dst.min())
    max_dst_idx = int(dataset.dst.max())
    evaluator = Evaluator(name=args.dataset)
    def get_memory_usage():
        if device.type == 'cuda':
            return torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        return 0.0
    for run in range(num_runs):
        seed_used = initial_seed + run
        torch.manual_seed(seed_used)
        logger.info(f"Starting run {run+1}/{num_runs} with seed {seed_used}")
        input_dim = 12 if params['big_hood'] else 4
        model = SequenceClassifier(input_dim=input_dim,
                                   d_model=params['d_model'],
                                   nhead=params['nhead'],
                                   num_layers=params['num_layers'],
                                   dim_feedforward=params['dim_feedforward'],
                                   dropout=params['dropout']).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.BCELoss()
        best_model_path = os.path.join(model_dir, f"best_model_run{run}.pth")
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        # load precalculated adjtable for train and val
        src_seg_train_cpu, dst_seg_train_cpu, ts_seg_train_cpu, adjtable_seg_train_cpu = \
            get_segment(dataset, 'train',
                        params['num_segment_train'], params['num_segment_val'],
                        params['num_segment_test'], params['num_neighbor'])
        src_seg_train, dst_seg_train, ts_seg_train, adjtable_seg_train = src_seg_train_cpu.to(device), dst_seg_train_cpu.to(device), ts_seg_train_cpu.to(device), adjtable_seg_train_cpu.to(device)
        dataset.load_val_ns()
        src_seg_val_cpu, dst_seg_val_cpu, ts_seg_val_cpu, adjtable_seg_val_cpu = \
            get_segment(dataset, 'val',
                        params['num_segment_train'], params['num_segment_val'],
                        params['num_segment_test'], params['num_neighbor'])
        src_seg_val, dst_seg_val, ts_seg_val, adjtable_seg_val = src_seg_val_cpu.to(device), dst_seg_val_cpu.to(device), ts_seg_val_cpu.to(device), adjtable_seg_val_cpu.to(device)
        epoch_times = []
        best_mrr = 0.
        for epoch in range(params['num_epochs']):
            epoch_start = time.time()
            train_loss_list = []
            # clear adjtable before each epoch
            src_seg_train[:], dst_seg_train[:], ts_seg_train[:], adjtable_seg_train[:] = src_seg_train_cpu, dst_seg_train_cpu, ts_seg_train_cpu, adjtable_seg_train_cpu
            for i in tqdm(range(src_seg_train.size(1)), desc=f"train {epoch+1}/{params['num_epochs']}"):
                src_i, dst_i, ts_i = get_batch(src_seg_train, dst_seg_train, ts_seg_train, i)
                dst_seg_neg = torch.randint(min_dst_idx, max_dst_idx,(src_i.size(0), 1), device=device)
                query_pos_first = torch.hstack([dst_i, dst_seg_neg])
                x, y = get_x_y(adjtable_seg_train, src_i, dst_i, query_pos_first, params['num_latest'], params['big_hood'])
                train_loss = train_one_batch(model, x, y , criterion, opt)
                train_loss_list.append(train_loss)
                
            epoch_elapsed = time.time() - epoch_start
            epoch_times.append(epoch_elapsed)
            
            train_loss = sum(train_loss_list)/len(train_loss_list)
            logger.info(f"Epoch {epoch+1}/{params['num_epochs']} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Time: {epoch_elapsed:.2f}s")
            ################################################################## val
            outputs_all = []
            val_loss_list = []
            src_seg_val[:], dst_seg_val[:], ts_seg_val[:], adjtable_seg_val[:] = src_seg_val_cpu, dst_seg_val_cpu, ts_seg_val_cpu, adjtable_seg_val_cpu
            for i in tqdm(range(src_seg_val.size(1)), desc='val'):
                src_i, dst_i, ts_i = get_batch(src_seg_val, dst_seg_val, ts_seg_val, i)
                query_pos_first = torch.hstack([dst_i, load_ns_eval(dataset, 'val', src_i, dst_i, ts_i)])
                x, y = get_x_y(adjtable_seg_val, src_i, dst_i, query_pos_first, params['num_latest'], params['big_hood'])
                batch_loss, outputs = evaluate(model, x, y, criterion)
                val_loss_list.append(batch_loss)
                outputs_all.append(outputs.view(-1, query_pos_first.size(1)).cpu())
            val_loss = sum(val_loss_list)/len(val_loss_list)
            outputs_all = np.vstack(outputs_all)
            input_dict = {"y_pred_pos": outputs_all[:, 0],
                          "y_pred_neg": outputs_all[:, 1:],
                          "eval_metric": ['mrr']}
            val_mrr = evaluator.eval(input_dict)['mrr']
            logger.info(f"Run {run+1} Val Results | "
                        f"Val loss: {val_loss:.4f} | "
                        f"Val mrr: {val_mrr:.4f}")
            if best_mrr < val_mrr:
                best_mrr = val_mrr
                torch.save(model.state_dict(), best_model_path)
                print(f"Val mrr: {val_mrr:.4f}")
            ##################################################################
        model.load_state_dict(torch.load(best_model_path))
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        max_gpu_mem = get_memory_usage()
        
        logger.info(f"Training completed | Avg epoch time: {avg_epoch_time:.2f}s | "
                    f"Max GPU mem: {max_gpu_mem:.2f}MB")
        # load precalculated adjtable for test
        src_seg_test_cpu, dst_seg_test_cpu, ts_seg_test_cpu, adjtable_seg_test_cpu = \
            get_segment(dataset, 'test',
                        params['num_segment_train'], params['num_segment_val'],
                        params['num_segment_test'], params['num_neighbor'])
        src_seg_test, dst_seg_test, ts_seg_test, adjtable_seg_test = src_seg_test_cpu.to(device), dst_seg_test_cpu.to(device), ts_seg_test_cpu.to(device), adjtable_seg_test_cpu.to(device)
        dataset.load_test_ns()
        outputs_all = []
        test_loss_list = []
        for i in tqdm(range(src_seg_test.size(1)), desc='test'):
            src_i, dst_i, ts_i = get_batch(src_seg_test, dst_seg_test, ts_seg_test, i)
            query_pos_first = torch.hstack([dst_i, load_ns_eval(dataset, 'test', src_i, dst_i, ts_i)])
            x, y = get_x_y(adjtable_seg_test, src_i, dst_i, query_pos_first, params['num_latest'], params['big_hood'])
            batch_loss, outputs = evaluate(model, x, y, criterion)
            test_loss_list.append(batch_loss)
            outputs_all.append(outputs.view(-1, query_pos_first.size(1)).cpu())
        test_loss = sum(test_loss_list)/len(test_loss_list)
        outputs_all = np.vstack(outputs_all)
        input_dict = {"y_pred_pos": outputs_all[:, 0],
                      "y_pred_neg": outputs_all[:, 1:],
                      "eval_metric": ['mrr']}
        test_mrr = evaluator.eval(input_dict)['mrr']
        logger.info(f"Run {run+1} Test Results | "
                    f"Loss: {test_loss:.4f} | "
                    f"Mrr: {test_mrr:.4f}")
        record_results(args.dataset, params, test_loss, test_mrr, seed_used, avg_epoch_time, max_gpu_mem)
if __name__ == "__main__":
    main()