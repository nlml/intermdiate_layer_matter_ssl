import time
import numpy as np
import cuml
from cuml.metrics import roc_auc_score


def load(name):
    return np.load(f"{name}_logits.npy"), np.load(f"{name}_y.npy")


train_logits, train_y = load("train")

lr = cuml.LogisticRegression(fit_intercept=True)
now = time.time()
lr.fit(train_logits, train_y)
print(time.time() - now)


def eval(lr, name):
    valid_logits, valid_y = load(name)
    valid_preds = lr.predict_proba(valid_logits)
    accu = (valid_y == valid_preds.argmax(1)).mean()
    roc = roc_auc_score(valid_y, valid_preds[:, 1])
    print(name, "accu", accu, "roc", roc)


eval(lr, "test")
eval(lr, "valid")
eval(lr, "train")
