try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix, classification_report
    from sklearn.metrics import accuracy_score, recall_score, classification_report
    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        f1_micro = f1_score(labels, preds, labels=[1, 2, 3], average='micro')
        f1_macro = f1_score(labels, preds, average='macro')
        f1_claim = f1_score(labels, preds, labels=[1], average='micro')
        f1_evidence = f1_score(labels, preds, labels=[2], average='micro')

        return {
            'eval_f1_micro': f1_micro,
            'eval_f1_macro': f1_macro,
            'f1_claim':f1_claim,
            'f1_evidence':f1_evidence,
        }

    def pico_scores(preds, labels):
        report = classification_report(labels, preds, labels=[1,2,3,4], target_names=['None', 'Intervention', 'Outcome', 'Population'] )
        return report

    def f1_scores(y_pred, y_true, labelfilter=None):

        f1_micro_filtered = f1_score(y_true, y_pred, labels=labelfilter, average='micro')
        f1_macro_filtered = f1_score(y_true, y_pred, labels=labelfilter, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        clf_report = classification_report(y_true, y_pred)

        return {
            "eval_f1_micro_filtered": f1_micro_filtered,
            "eval_f1_macro_filtered": f1_macro_filtered,
            'eval_f1_micro': f1_micro,
            'eval_f1_macro': f1_macro,
            'clf_report': clf_report
        }


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def compute_confusion_matrix(task_name, y_pred, y_true):

        assert len(y_pred) == len(y_true)
        if task_name == "multichoice" or task_name == "relclass":
            return confusion_matrix(y_true, y_pred)
        else:
            raise KeyError(task_name)

    def compute_metrics(task_name, y_pred, y_true):
        assert len(y_pred) == len(y_true)
        if task_name == "seqtag":
            return acc_and_f1(y_pred, y_true)
        elif task_name == "picoseqtag":
            return pico_scores(y_pred, y_true)
        elif task_name == "relclass":
            return f1_scores(y_pred, y_true, [0, 1])
        elif task_name == "outcomeclf":
            return f1_scores(y_pred, y_true)
        elif task_name == "multichoice":
            return f1_scores(y_pred, y_true, [0, 1])
        else:
            raise KeyError(task_name)

