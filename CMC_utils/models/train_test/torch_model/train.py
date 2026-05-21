import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, Tuple, Optional, Dict
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
import pprint
import torch.nn.functional as F

from sklearn.utils.class_weight import compute_class_weight 

from .outputs_functions import *
from CMC_utils.save_load import create_directory
from CMC_utils.models import initialize_weights
from CMC_utils.metrics import metrics_computation_df
from CMC_utils.datasets import SupervisedTabularDatasetTorch
from CMC_utils.miscellaneous import do_really_nothing, seed_worker

log = logging.getLogger(__name__)

__all__ = ["train_torch_model"]

def train_torch_model(model: torch.nn.Module, train_set: SupervisedTabularDatasetTorch, model_params: dict, model_path: str, val_set: SupervisedTabularDatasetTorch, train_params: Union[dict, DictConfig], test_fold: int = 0, val_fold: int = 0, use_weights: bool = True, **kwargs) -> None:
    
    # 1. DataLoader 初始化
    train_dataloader = DataLoader(train_set, batch_size=train_params.dl_params.batch_size, shuffle=True, drop_last=True, worker_init_fn=seed_worker)
    val_dataloader = DataLoader(val_set, batch_size=train_params.dl_params.batch_size, shuffle=False, drop_last=False)
    dataloaders = dict(train=train_dataloader, val=val_dataloader)
 
    # 2. 模型权重初始化
    initialize_weights(model, train_params.initializer)
 
    # 3. 训练管理器 (Training Manager) 初始化
    metrics = train_params.get("set_metrics", {})
    model_path = os.path.join(model_path, model_params["name"])
    create_directory(model_path)
    filename = f"{test_fold}_{val_fold}"
    
    tr_manager = instantiate(train_params["manager"], model, filename=filename, path=model_path, metrics=metrics, optimizer=train_params["optimizer"], model_params=model_params, _recursive_=False)
 
    model = tr_manager.load_checkpoint(model)
    model = model.to(tr_manager.device)
 
    callback_options = {True: tr_manager.callbacks, False: do_really_nothing}
    print_options = {0: do_really_nothing, 1: log.info}
 
    epoch, phase, performance = -1, None, None
 
    if tr_manager.early_stop.state["early_stop"]:
        return
 
    # ==========================================
    # 4. 类别权重计算 (Class Weights Calculation)
    # ==========================================
    log.info("--- Applying Label-Aware Sample Weights ---")
    try:
        y_train_labels = train_set.labels
        log.info("Successfully accessed labels via `train_set.labels`.")
    except AttributeError:
        log.warning("Attribute `train_set.labels` not found. Falling back to iterating through the dataset. This might be slow.")
        y_train_labels = np.array([label for _, label, _ in train_set])
        
    if isinstance(y_train_labels, pd.Series):
        y_train_labels = y_train_labels.values
    elif isinstance(y_train_labels, torch.Tensor):
        y_train_labels = y_train_labels.cpu().numpy()
 
    if y_train_labels.ndim > 1:
        y_train_labels = np.argmax(y_train_labels, axis=1)
 
    class_labels = np.unique(y_train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(tr_manager.device)

    log.info(f"Computed class weights: {class_weights}")
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor).to(tr_manager.device)
    criterions = [loss_fn]
    
    # ==========================================
    # 5. 正则化模块初始化 (Regularizers Init)
    # ==========================================
    traditional_regs_configs = {name: params for name, params in train_params.regularizer.items() if name != 'iam'}
    regularizers = [instantiate(params["init_params"]).to(tr_manager.device) for params in traditional_regs_configs.values()]
    
    # --- IAM 模块初始化 ---
    iam_reg = None
    iam_reg_config = train_params.regularizer.get("iam", None) 
    
    if iam_reg_config:
        log.info("Initializing Imbalance-Aware Masking (IAM-Reg)...")
        # Hydra 的 instantiate 会自动处理 .yaml 中的 total_epochs 插值
        iam_reg = instantiate(iam_reg_config.init_params).to(tr_manager.device)
        log.info(f"IAM-Reg initialized. Dynamic mode: {hasattr(iam_reg, 'set_epoch')}")
        # 打印部分配置以确认
        if hasattr(iam_reg_config.init_params, 'min_p_majority'):
            log.info(f"IAM Params: Majority[{iam_reg_config.init_params.min_p_majority}-{iam_reg_config.init_params.max_p_majority}]")

    # ==========================================
    # 6. 训练循环 (Training Loop)
    # ==========================================
    for epoch in tr_manager.epochs():
 
        if tr_manager.early_stop.state["early_stop"]:
            break
            
        # [NEW] >>> 动态更新遮蔽率 <<<
        # 检查 iam_reg 是否存在且是否有 set_epoch 方法
        if iam_reg is not None and hasattr(iam_reg, 'set_epoch'):
            iam_reg.set_epoch(epoch)
            # 可选：打印当前的遮蔽率用于调试 (debug level)
            # log.debug(f"Epoch {epoch}: IAM majority_p={iam_reg.current_p_maj:.3f}")

        optimizer = tr_manager.optimizer
        
        for phase, dataloader in dataloaders.items():
            
            # 调用预测/训练函数
            model, epoch_loss, epoch_results = model_predict(
                model=model, 
                dataloader=dataloader, 
                criterions=criterions, 
                regularizers=regularizers,
                traditional_regs_configs=traditional_regs_configs,
                optimizer=optimizer, 
                tr_manager=tr_manager, 
                train_params=train_params, 
                phase=phase, 
                iam_reg=iam_reg, # 传入 IAM 实例
                **kwargs
            )
 
            # 指标计算
            epoch_performance = metrics_computation_df(epoch_results, metrics=metrics, classes=dataloader.dataset.classes, use_weights=True, verbose=0)
 
            print_options[tr_manager.verbose]('{} loss: {:.4f} '.format(phase, epoch_loss) + " ".join([f"{metric}: {epoch_performance[metric].values.tolist()}" for metric in epoch_performance.columns]))
 
            tr_manager.history[phase]['loss'].append(epoch_loss)
            for metric in epoch_performance.columns:
                performance = epoch_performance[metric].values.tolist()
                tr_manager.history[phase][metric].append(performance)
                performance = np.mean(performance)
 
            callback_options[phase == "val"](model, epoch, epoch_loss, performance)
 
    if phase is not None:
        tr_manager.save_checkpoint(epoch + (not tr_manager.early_stop.state["early_stop"]))
        log.info("Model trained")
        print_options[tr_manager.verbose and phase == "val"](tr_manager.model_checkpoint.state)

def model_predict(model: torch.nn.Module, dataloader: DataLoader, criterions: list, regularizers: list, traditional_regs_configs: Dict, optimizer, tr_manager, train_params: DictConfig, phase: str = "val", iam_reg: Optional[torch.nn.Module] = None, **kwargs) -> Tuple[torch.nn.Module, float, pd.DataFrame]:
    
    output_to_pred_options = {"binary": surpass_threshold, "categorical": max_index}
    model_options = {True: model.train, False: model.eval}
    train_params_options = {True: tr_manager.update_train_params, False: do_really_nothing}
    
    # IAM 状态切换：如果有 IAM 模块，跟随训练/验证模式切换
    iam_reg_options = {True: (iam_reg.train if iam_reg else do_really_nothing), False: (iam_reg.eval if iam_reg else do_really_nothing)}

    running_loss = 0.0
    total_results = pd.DataFrame()

    model_options[phase == "train"]()
    iam_reg_options[phase == "train"]() # 确保 IAM 在 eval 模式下不进行遮蔽

    pbar = tqdm(dataloader, leave=False, disable=(not train_params.dl_params.verbose_batch))
    with torch.set_grad_enabled(phase == 'train'):
        for *input_list, labels, idxs in pbar:
            input_list = [inputs.float().to(tr_manager.device) for inputs in input_list]
            labels = labels.to(tr_manager.device)

            optimizer.zero_grad()
            
            # [NEW] 应用 IAM 动态遮蔽
            # 只在训练阶段且 iam_reg 存在时执行
            if iam_reg is not None and phase == 'train':
                # 注意：这里不需要调用 .cuda() 或 .to()，因为在 train_torch_model 里已经 to(device) 了
                # 传入原始特征和标签，获取遮蔽后的特征
                original_features = input_list[0].clone()
                masked_features = iam_reg(original_features, labels)
                input_list[0] = masked_features

            outputs = model(*[inputs.float() for inputs in input_list])

            if labels.ndim > 1 and labels.shape[-1] > 1:
                processed_labels = labels.argmax(dim=1)
            else:
                processed_labels = labels.squeeze() 
            
            loss = criterions[0](outputs, processed_labels.long())

            if regularizers:
                for regularizer_params, regularizer in zip(traditional_regs_configs.values(), regularizers):
                    loss += regularizer_params.alpha * regularizer(model)

            train_params_options[phase == "train"](loss, optimizer)

            # --- 后续处理保持不变 ---
            if (dataloader.dataset.label_type == "binary" and outputs.shape[-1] == 2) or dataloader.dataset.label_type == "categorical":
                outputs = F.softmax(outputs, dim=-1)

            preds = output_to_pred_options[dataloader.dataset.label_type](outputs, return_first=True, **kwargs)

            if labels.shape != preds.shape:
                labels = output_to_pred_options[dataloader.dataset.label_type](labels, return_first=True, **kwargs)
            
            idxs = np.array(idxs)
            labels = np.squeeze(labels.cpu().detach().numpy()).astype(int)
            preds = np.squeeze(preds.cpu().detach().numpy()).astype(int)
            outputs = outputs.cpu().detach().numpy().astype(float)

            if dataloader.dataset.label_type == "binary":
                if outputs.shape[1] == 2:
                    outputs = outputs[:, 1]
                outputs = np.squeeze(outputs)

            batch_results = pd.DataFrame(dict( ID=idxs, label=labels.tolist(), prediction=preds.tolist(), probability=outputs.tolist()))
            total_results = pd.concat( [total_results, batch_results], axis=0, ignore_index=True)

            running_loss += loss.item() * input_list[0].size(0)

            pbar.set_postfix_str( "loss {:.4f}, ".format(running_loss / total_results.shape[0]) )

    running_loss = running_loss / total_results.shape[0]

    return model, running_loss, total_results

if __name__ == "__main__":
    pass