from typing import Union
from hydra.utils import call, get_method
from omegaconf import DictConfig, OmegaConf

# 把新函数名加到这里
__all__ = ["set_metrics_params", "do_nothing"]

# 这是我们新增的“什么都不做”的函数
def do_nothing(metric_params, **kwargs):
    """
    A placeholder function that does nothing and returns the metric parameters as is.
    Used for metrics that do not require special parameter setting.
    This version now prepares the 'init' key.
    """
    # 关键改动：构建 'init' 字典
    # 使用 metric_params['function'] 作为 _target_
    # 使用 metric_params['params'] 作为参数
    metric_params['init'] = {
        '_target_': metric_params['function'],
        **metric_params.get('params', {})
    }
    return metric_params
 
 
def set_metrics_params(metrics: Union[dict, DictConfig], preprocessing_params: dict):
    """
    Set the parameters of the metrics.
    Now this function restructures each metric to include an 'init' key
    for instantiation by hydra.utils.call.
    """
    
    # ========================== START: 修改这里的代码 ==========================
    # 只有当传入的是 DictConfig 对象时，才解除结构化限制
    if isinstance(metrics, DictConfig):
        OmegaConf.set_struct(metrics, False)
    # ==========================  END: 修改这里的代码  ==========================
 
    processed_metrics = {}
    for key, metric_params in metrics.items():
        param_setter_func = get_method(metric_params["set_params_function"])
        updated_metric_params = param_setter_func(metric_params, preprocessing_params=preprocessing_params)
        processed_metrics[key] = updated_metric_params
 
    return processed_metrics


if __name__ == "__main__":
    pass
