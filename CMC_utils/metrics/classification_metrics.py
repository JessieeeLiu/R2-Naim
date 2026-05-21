from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
 

__all__ = ["AUC", "accuracy", "recall", "precision", "f1", "macro_f1", "g_mean"]
 
 
def AUC(y_true, y_score, average='macro', sample_weight=None, max_fpr=None, multi_class='ovr', labels=None, **kwargs):
    """Compute Area Under the Curve (AUC) from prediction scores"""
    return roc_auc_score(y_true, y_score, average=average, sample_weight=sample_weight, max_fpr=max_fpr,
                         multi_class=multi_class, labels=labels)
 
 
def accuracy(y_true, y_pred, normalize=True, sample_weight=None, **kwargs):
    """Accuracy classification score. (不推荐用于不均衡问题)"""
    return accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight) 
 
 
def recall(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn',
           **kwargs):
    """Compute the recall (通常指二分类的正类召回率)"""
    return recall_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average,
                        sample_weight=sample_weight, zero_division=zero_division)
 
 
def precision(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn', **kwargs):
    """Compute the precision (通常指二分类的正类精确度)"""
    return precision_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average,
                           sample_weight=sample_weight, zero_division=zero_division)
 
 
def f1(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn', **kwargs):
    """Compute the F1 score (通常指二分类的正类F1)"""
    return f1_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight,
                    zero_division=zero_division)
 
# ==============================================================================
# =================== 新增的核心评估指标 (START) =================================
# ==============================================================================
 
def macro_f1(y_true, y_pred, labels=None, sample_weight=None, zero_division='warn', **kwargs):
    """
    计算宏平均F1分数 (Macro F1-Score)。
    这是评估不均衡分类性能的关键指标，它平等对待每个类别。
    """
    # 强制 average='macro'
    return f1_score(y_true, y_pred, labels=labels, average='macro', sample_weight=sample_weight,
                    zero_division=zero_division)
 
 
def g_mean(y_true, y_pred, labels=None, sample_weight=None, **kwargs):
    """
    计算几何平均值 (Geometric Mean Score)。
    这是另一个评估不均衡分类性能的关键指标，衡量模型在所有类别上的平衡表现。
    """
    # imblearn的g-mean函数会自动处理
    return geometric_mean_score(y_true, y_pred, labels=labels, sample_weight=sample_weight)
 
# ==============================================================================
# =================== 新增的核心评估指标 (END) ===================================
# ==============================================================================
 
 
if __name__ == "__main__":
    # 示例用法
    y_true = [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]  # 严重不均衡
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 一个只预测多数类的糟糕模型
    
    print(f"Accuracy: {accuracy(y_true, y_pred)}") # 准确率很高，产生误导
    print(f"Binary F1 (for class 1): {f1(y_true, y_pred)}") # F1很低
    print(f"Macro F1: {macro_f1(y_true, y_pred)}") # Macro F1能反映出问题
    print(f"G-Mean: {g_mean(y_true, y_pred)}")     # G-Mean也能反映出问题