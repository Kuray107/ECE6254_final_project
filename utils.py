import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

class SaveBest:
    """ Callback to get the best value and epoch
    Args:
        val_comp: str, (Default value = "inf") "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
    Attributes:
        val_comp: str, "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
        best_val: float, the best values of the model based on the criterion chosen
        best_epoch: int, the epoch when the model was the best
        current_epoch: int, the current epoch of the model
    """
    def __init__(self, val_comp="inf"):
        self.comp = val_comp
        if val_comp in ["inf", "lt", "desc"]:
            self.best_val = np.inf
        elif val_comp in ["sup", "gt", "asc"]:
            self.best_val = 0
        else:
            raise NotImplementedError("value comparison is only 'inf' or 'sup'")
        self.best_epoch = 0
        self.current_epoch = 0

    def apply(self, value):
        """ Apply the callback
        Args:
            value: float, the value of the metric followed
        """
        decision = False
        if self.current_epoch == 0:
            decision = True
        if (self.comp == "inf" and value < self.best_val) or (self.comp == "sup" and value > self.best_val):
            self.best_epoch = self.current_epoch
            self.best_val = value
            decision = True
        self.current_epoch += 1
        return decision


def calculate_F1_score(preds, labels):
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    F1_score = f1_score(labels.cpu(), preds.cpu())
    acc = (tp+tn) / (tp+tn+fp+fn)

    return F1_score, precision, recall, acc

def get_auc_score(preds, labels):
    auc_score = roc_auc_score(labels, preds)
    return auc_score

def delete_outlier(np_list):
    if len(np_list) < 3:
        return np_list
    else:
        np_list = np.delete(np_list, np_list.argmin())
        np_list = np.delete(np_list, np_list.argmax())
        return np_list
