from common.dataloader import get_dataloaders, load_dataset
from common.utils import load_config, set_logger, print_to_json
from common import data_preprocess
import logging
import argparse
import torch
import numpy as np
from networks.anomaly_transformer.solver import AnomalyTransformer

# 假设你已经有了params字典，可以通过load_config加载
# 例如：
# parser = argparse.ArgumentParser()
# parser.add_argument("--config", type=str, default="./benchmark_config/", help="The config directory.")
# parser.add_argument("--expid", type=str, default="anomaly_transformer_SMD")
# parser.add_argument("--gpu", type=int, default=-1)
# args = vars(parser.parse_args())
# config_dir = args["config"]
# experiment_id = args["expid"]
# params = load_config(config_dir, experiment_id)
# set_logger(params, args)
# logging.info(print_to_json(params))

# 假设params已经定义
params = {
    "dataset_id": "SWAT",
    "model_id": "anomaly_transformer",
    "exp_id": "anomaly_transformer_swat",

    "data_root": "../data/SWAT/",
    "entities": ["swat"],
    "dim": 19,  # 假设数据维度
    "valid_ratio": 0,
    "test_label_postfix": "test_label3.pkl",
    "test_postfix": "test3.pkl",
    "train_postfix1": "train1.pkl",
    "train_postfix2": "train2.pkl",
    "nrows": None,
    "normalize": "minmax",
    "window_size": 30,  # 假设窗口大小
    "stride": 1,
    "batch_size": 128,
    "num_workers": 1,
    "lr": 0.0001,
    "nb_epoch": 10,
    "k": 3,
    "model_root": "../model_save/",  # 模型保存路径
    "device": "4" if torch.cuda.is_available() else "cpu",
    "eval": {}  # 评估参数，这里简化
}

# 添加日志配置
set_logger(params, {})  # 传入空的args字典
logging.info(print_to_json(params))

# 加载数据集
data_dict = load_dataset(
    data_root=params["data_root"],
    entities=params["entities"],
    dim=params["dim"],
    valid_ratio=params["valid_ratio"],
    test_label_postfix=params["test_label_postfix"],
    test_postfix=params["test_postfix"],
    train_postfix=params["train_postfix1"],
    nrows=params["nrows"],
)

data_dict2 = load_dataset(
    data_root=params["data_root"],
    entities=params["entities"],
    dim=params["dim"],
    valid_ratio=params["valid_ratio"],
    test_label_postfix=params["test_label_postfix"],
    test_postfix=params["test_postfix"],
    train_postfix=params["train_postfix2"],
    nrows=params["nrows"],
)

# # 预处理
# pp = data_preprocess.preprocessor(model_root=params["model_root"])
# data_dict = pp.normalize(data_dict, method=params["normalize"])

# 滑动窗口
window_dict = data_preprocess.generate_windows(
    data_dict,
    window_size=params["window_size"],
    stride=params["stride"],
)

window_dict2 = data_preprocess.generate_windows(
    data_dict2,
    window_size=params["window_size"],
    stride=params["stride"],
)

entity = params["entities"][0]  # 假设只处理一个实体
windows = window_dict[entity]
windows2 = window_dict2[entity]
train_windows = np.concatenate((windows["train_windows"], windows2["train_windows"]), axis=0)
test_windows = windows["test_windows"]  # 这里只是为了get_dataloaders，实际训练不需要test_windows
test_windows_label = windows["test_label"]  # 这里只是为了get_dataloaders，实际训练不需要test_windows_label

train_loader, valid_loader, _ = get_dataloaders(
    train_windows,
    test_windows,  # 这里可以传入一个空的或占位的test_windows，因为train_model只用到train_loader和valid_loader
    batch_size=params["batch_size"],
    num_workers=params["num_workers"],
)

model = AnomalyTransformer(
    lr=params["lr"],
    num_epochs=params["nb_epoch"],
    k=params["k"],
    win_size=params["window_size"],
    input_c=params["dim"],
    output_c=params["dim"],
    batch_size=params["batch_size"],
    model_save_path=params["model_root"],
    device=params["device"],
)

# 训练模型
trained_model = model.fit(train_loader, valid_loader)
print("模型训练完成并保存到:", params["model_root"])
