import time
from math import exp
from networks.anomaly_transformer import AnomalyTransformer
import torch
import numpy as np
from common.dataloader import get_dataloaders, load_dataset
from common import data_preprocess
from common.score_show import plot_anomaly_scores  # 添加导入语句
from common.utils import load_config, set_logger, print_to_json
import logging

# 假设你已经训练好模型并保存了checkpoint.pth
# 并且params字典与训练时一致
params = {
    "dataset_id": "SWAT",
    "model_id": "anomaly_transformer",
    "exp_id": "anomaly_transformer_swat",

    "data_root": "../data/SWAT/",
    "entities": ["swat"],
    "dim": 19,  # 假设数据维度
    "valid_ratio": 0,
    "test_label_postfix": "test_label3.pkl",
    "test_postfix": "test.pkl",
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

# 初始化模型实例
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

# 滑动窗口
window_dict = data_preprocess.generate_windows(
    data_dict,
    window_size=params["window_size"],
    stride=params["stride"],
)
test_windows = window_dict[params["entities"][0]]["test_windows"]

# 模拟流式单时间点输入数据
result = []
start_time = time.time()
for i in range(test_windows.shape[0]):
    # 将numpy数组转换为torch张量，并调整形状
    single_point_data = torch.from_numpy(test_windows[i]).float().unsqueeze(0)
    result.append(model.predict_single_point(single_point_data))
end_time = time.time()
print(f"Time taken for {test_windows.shape[0]} predictions: {end_time - start_time} seconds")

# 为了演示，我将使用您提供的 anomaly_range 和一个默认的 gap_time
anomaly_range_example = [[2142, 2201], [3043, 3102], [3613, 3672], [4514, 4573], [4795, 4854]]
gap_time_example = 1  # 假设每个时间步长为1

plot_anomaly_scores(
    result,
    0,  # test_start 假设从0开始
    len(result),  # test_end 假设为异常得分的长度
    np.max(result),  # valid_anomaly_max 可以是异常得分的最大值
    0.5,  # alpha 阈值乘数，可以根据需要调整
    anomaly_range_example,
    gap_time_example,
    params["exp_id"].rsplit('_', 1)[0]  # 使用实验ID中最后一个下划线之前的部分作为模型名称
)
