"""
Validation Metrics and Plots
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_precision_recall
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, precision_recall_curve, auc)
from sklearn.preprocessing import label_binarize

__all__ = ['plot_classification_metrics']


def softmax(logits):
    """ Softmax function

    Parameters
    ----------
    logits : np.ndarray
        shape=(n_samples, n_classes)

    Returns
    -------
    softmax : np.ndarray
        shape=(n_samples, n_classes)

        (softmax.sum(axis=1) == 1).all() == True

    Examples
    ------
    >>> softmax([1, 1])
    array([[0.5, 0.5]])
    """
    if not isinstance(logits, np.ndarray):
        logits = np.array(logits)
    if logits.ndim == 1:  # shape = (n_classes,)
        logits = logits.reshape((1, -1))  # shape = (1,n_classes)
    shift = logits.max(axis=1, keepdims=True)
    exp_x = np.exp(logits - shift)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def plot_classification_metrics(y_true,
                                y_pred_proba,
                                classes_to_plot=None,
                                threshold=None,
                                plot_micro=True):
    """ Plot ROC Curve, Precision-Recall Curve and Confusion Matrix

    Parameters
    ----------
    y_true : array-like, shape (n_samples)
        Ground truth labels.

    y_pred_proba : array-like, shape (n_samples, n_classes)
        Prediction probabilities or decision scores for each class
        returned by a classifier.

    classes_to_plot : list-like, optional
        Classes for which the ROC curve should be plotted. e.g. [0, 'cold'].
        If the specified class does not exist, it will be ignored.
        If ``None``, all classes will be plotted.

    threshold : None or float
        if a float is set, it will be used as the decision threshold for
        binary classification

    plot_micro : bool
        whether to plot the averaged roc_curve and the precision-recall curve
        using average method 'micro'

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure` object

    axs : Axes object or array of Axes objects.
    """

    fig = plt.figure(dpi=100, figsize=(10.5, 8))
    ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)
    ax3 = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=2)

    # region Plot ROC Curve
    plot_roc(
        y_true,
        y_pred_proba,
        plot_macro=False,
        plot_micro=plot_micro,
        classes_to_plot=classes_to_plot,
        ax=ax1)

    # endregion

    # region plot Precision-Recall Curve
    plot_precision_recall(
        y_true,
        y_pred_proba,
        plot_micro=plot_micro,
        classes_to_plot=classes_to_plot,
        ax=ax2)
    ax2.legend(loc='lower right')
    # endregion

    # region Plot Confusion Matrix
    y_pred_idx = np.argmax(y_pred_proba, axis=-1)
    labels = np.sort(np.unique(y_true))
    y_pred = labels[y_pred_idx]
    plot_confusion_matrix(y_true, y_pred, normalize=True, ax=ax3)
    # endregion
    axs = [ax1, ax2, ax3]

    if threshold:
        # region Plot Confusion Matrix
        labels = np.sort(np.unique(y_true))
        assert len(labels) == 2, """Problem is not binary classification
        but decision threshold is set"""
        ax4 = plt.subplot2grid((4, 4), (2, 2), rowspan=2, colspan=2)
        is_positive = y_pred_proba[:, 1] > threshold
        y_pred = labels[is_positive.astype('int')]
        plot_confusion_matrix(y_true, y_pred, normalize=True, ax=ax4)
        ax4.set_title('Confusion Matrix with adjusted '
                      'decision threshold: {:.2}'.format(threshold))

        # update color limit
        im3 = ax3.get_images()[0]
        clim = im3.get_clim()
        im4 = ax4.get_images()[0]
        im4.set_clim(clim)
        axs.append(ax4)

        # endregion
    fig.tight_layout()
    return fig, axs


class ClassificationReport:
    def __init__(self):
        self.labels_ = None
        self.is_binary_ = None
        self.n_classes_ = None

    def micro_roc_auc(self, y_true, y_pred_proba):
        """Calculate the micro average AUC score"""

        # transform multi-class labels to one-hot labels
        # and then treat the problem as n_lables binary classifications
        # using one-vs-all approach
        binarized_y_true = label_binarize(y_true, classes=self.labels_)
        if self.n_classes_ == 2:
            binarized_y_true = np.hstack((1 - binarized_y_true,
                                          binarized_y_true))
        fpr, tpr, _ = roc_curve(binarized_y_true.ravel(), y_pred_proba.ravel())
        micro_roc_auc = auc(fpr, tpr)
        return micro_roc_auc

    def micro_average_precision_score(self, y_true, y_pred_proba):
        """Calculate the micro average precision score"""

        # transform multi-class labels to one-hot labels
        # and then treat the problem as n_lables binary classifications
        # using one-vs-all approach

        binarized_y_true = label_binarize(y_true, classes=self.labels_)

        if self.n_classes_ == 2:
            binarized_y_true = np.hstack((1 - binarized_y_true,
                                          binarized_y_true))
        precision, recall, _ = precision_recall_curve(binarized_y_true.ravel(),
                                                      y_pred_proba.ravel())
        average_precision_score = auc(recall, precision)
        return average_precision_score

    def _report_for_pos_label(self, y_test, y_pred_proba, pos_label,
                              threshold):
        """Evaluation report for the specified positive label

        Only works for binary classification
        """
        idx = self.labels_.index(pos_label)
        decision_score = y_pred_proba[:, idx]
        y_pred = decision_score > threshold
        y_true = y_test == pos_label

        report = {}
        report['accuracy'] = accuracy_score(y_true, y_pred)
        fpr_, tpr_, _ = roc_curve(y_true, decision_score, pos_label=True)
        report['roc_auc'] = auc(fpr_, tpr_)
        precision_, recall_, _ = precision_recall_curve(
            y_true, decision_score, pos_label=True)
        report['average_precision_score'] = auc(recall_, precision_)
        report['precision'] = precision_score(
            y_true, y_pred, pos_label=True, average='binary')
        report['recall'] = recall_score(
            y_true, y_pred, pos_label=True, average='binary')
        report['f1_score'] = f1_score(
            y_true, y_pred, pos_label=True, average='binary')

        report = pd.DataFrame(report, index=[pos_label])

        report.loc['micro', 'precision'] = \
            precision_score(y_true, y_pred, pos_label=None, average='micro')
        report.loc['micro', 'recall'] = \
            recall_score(y_true, y_pred, pos_label=None, average='micro')
        report.loc['micro', 'f1_score'] = \
            f1_score(y_true, y_pred, pos_label=None, average='micro')
        return report

    def _binary_classification_report(self,
                                      y_test,
                                      y_pred_proba,
                                      positive_label=None,
                                      threshold=0.5):
        """ Evaluation report for binary classification"""

        if positive_label:  # only report for positive label
            report = self._report_for_pos_label(y_test, y_pred_proba,
                                                positive_label, threshold)
        else:  # report for each label
            report = pd.concat([
                self._report_for_pos_label(y_test, y_pred_proba, label,
                                           threshold) for label in self.labels_
            ],
                axis=0)
        report.loc['micro', 'roc_auc'] = \
            self.micro_roc_auc(y_test, y_pred_proba)
        report.loc['micro', 'average_precision_score'] = \
            self.micro_average_precision_score(y_test, y_pred_proba)
        return report

    def _multiclass_classification_report(self, y_test, y_pred, y_pred_proba):
        labels = self.labels_
        report = pd.DataFrame({
            'accuracy':
                accuracy_score(y_test, y_pred),
            'precision':
                precision_score(y_test, y_pred, pos_label=None, average=None),
            'recall':
                recall_score(y_test, y_pred, pos_label=None, average=None),
            'f1_score':
                f1_score(y_test, y_pred, pos_label=None, average=None),
        },
            index=labels)

        report.loc['micro', 'precision'] = precision_score(
            y_test, y_pred, pos_label=None, average='micro')
        report.loc['micro', 'recall'] = recall_score(
            y_test, y_pred, pos_label=None, average='micro')
        report.loc['micro', 'f1_score'] = f1_score(
            y_test, y_pred, pos_label=None, average='micro')

        aucs = []
        average_precision_scores = []
        for label in labels:
            idx = labels.index(label)
            decision_score = y_pred_proba[:, idx]
            fpr, tpr, _ = roc_curve(y_test, decision_score, pos_label=label)
            aucs.append(auc(tpr, fpr))
            precision, recall, _ = precision_recall_curve(
                y_test, decision_score, pos_label=label)
            average_precision_scores.append(auc(recall, precision))

        report.loc[labels, 'auc'] = aucs
        report.loc[labels,
                   'average_precision_score'] = average_precision_scores

        report.loc['micro', 'auc'] = self.micro_roc_auc(y_test, y_pred_proba)
        report.loc['micro', 'average_precision_score'] = \
            self.micro_average_precision_score(y_test, y_pred_proba)

        return report

    def classifiction_report(self,
                             classifier,
                             X_test,
                             y_test,
                             positive_label=None,
                             threshold=0.5):
        """ Return a classification report

        Parameters
        ----------
        classifier : Classifier
            must have a `predict_proba` or `decision_function` method

        X_test : array-like, shape=(n_samples, n_features)
            test feature matrix,

        y_test : array-like
            the ground truth labels

        positive_label : int, str or None, default None
            If not None, only the specified positive_label
            will be treated as positive. All the other labels
            are treated as negative.

            If None, one at at a time, each label will be
            treated as positive. Thus, metrics like AUC will be
            reported for each label.

        threshold : float
            If the predicted probability score of the positive class
            is larger than the threshold, it will be classified to the class
            specified by positive_label. This is only used for adjusting
            binary classification

        Returns
        -------
        report : pd.DataFrame
            a report of various metrics for evaluating the classifier
        """
        labels = sorted(set(y_test))
        n_classes = len(labels)
        is_binary = n_classes <= 2
        self.labels_ = labels
        self.n_classes_ = n_classes
        self.is_binary_ = is_binary

        if hasattr(classifier, 'predict_proba'):
            y_pred_proba = classifier.predict_proba(X_test)
        elif hasattr(classifier, 'decision_function'):
            decision_score = classifier.decision_function(X_test)
            y_pred_proba = softmax(decision_score)
        else:
            raise Exception('The classifier does not have'
                            'predict_proba or decision_function method')

        if is_binary:
            report = self._binary_classification_report(
                y_test, y_pred_proba, positive_label, threshold)
        else:
            y_pred = classifier.predict(X_test)
            report = self._multiclass_classification_report(
                y_test, y_pred, y_pred_proba)
        return report
