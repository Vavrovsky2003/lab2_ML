import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *

dd_mm = '29-01'
K = int(dd_mm[-2:]) % 4

d = 0.02

"""## 1."""

df = pd.read_csv('KM-12-1.csv')
print(df.head())

"""## 2."""

print(df['GT'].value_counts())

"""## 3.

### a.
"""

def get_metrics(df, td):
  train = df['GT']
  threshold_arange = np.arange(0, 1+td, td)
  results_list = []
  for threshold in threshold_arange:
      for model in ['Model_1', 'Model_2']:
          predictions = (df[model] > threshold).astype(int)

          accuracy = accuracy_score(train, predictions)
          precision = precision_score(train, predictions)
          recall = recall_score(train, predictions)
          f_score = f1_score(train, predictions)
          mcc = matthews_corrcoef(train, predictions)
          balanced_acc = balanced_accuracy_score(train, predictions)
          results_list.append({
              'Threshold': threshold,
              'Num': predictions.sum(),
              'Model': model,
              'Accuracy': accuracy,
              'Precision': precision,
              'Recall': recall,
              'F1-Score': f_score,
              'MCC': mcc,
              'Balanced Accuracy': balanced_acc,
              'Youden step': recall_score(train, predictions) + recall_score(train, predictions, pos_label=0) - 1
          })
  results_df = pd.DataFrame(results_list)
  constants_list = []
  for model in ['Model_1', 'Model_2']:
    auc_roc = roc_auc_score(df['GT'], df[model])

    precision, recall, thresholds = precision_recall_curve(df['GT'], df[model], pos_label=1)
    auc_prc = auc(recall, precision)
    constants_list.append({
        'Model': model,
        'AUC for ROC': auc_roc,
        'AUC for PRC': auc_prc,
        'Youden': np.max(results_df[results_df['Model'] == model]['Youden step'])
        })
  constants_df = pd.DataFrame(constants_list)
  return results_df, constants_df

metrics_df, constants_df = get_metrics(df, d)

"""### b."""

for model_name in ['Model_1', 'Model_2']:
  fig, ax = plt.subplots()
  metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'Balanced Accuracy']
  xs = metrics_df[metrics_df['Model'] == model_name]['Threshold']
  for i in metrics:
    ys = metrics_df[metrics_df['Model'] == model_name][i]
    plt.plot(xs, ys, label=i)
    plt.plot([0, 1], [np.max(ys)]*2, '--')
  metrics = ['AUC for ROC', 'AUC for PRC', 'Youden']
  for i in metrics:
    ys = constants_df[constants_df['Model'] == model_name][i]
    plt.plot([0, 1], [ys]*2, label=i)
  ax.set_xlabel('Threshold')
  ax.set_ylabel('Metric values')
  plt.legend()
  plt.show()

"""### c."""

for model_name in ['Model_1', 'Model_2']:
  fig, ax = plt.subplots()
  metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'Balanced Accuracy']
  xs = metrics_df[metrics_df['Model'] == model_name]['Num']
  for i in metrics:
    ys = metrics_df[metrics_df['Model'] == model_name][i]
    plt.plot(xs, ys, label=i)
    plt.plot([0, 1], [np.max(ys)]*2, '--')
    plt.axvline(xs.iloc[np.argmax(ys)], linestyle='--')

  ax.set_xlabel('Num')
  ax.set_ylabel('Metric values')
  plt.legend()
  plt.show()

"""### d."""

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, brentq

def intrsection(x0, y0, x1, y1):
  f0 = interp1d(x0, y0)
  f1 = interp1d(x1, y1)

  x = brentq(lambda x: f0(x) - f1(x), 0, 1)

  return x, f1(x)

for model_name in ['Model_1', 'Model_2']:
  figure, (pr_plot, roc_plot) = plt.subplots(ncols = 2, figsize=(15, 5))

  precision, recall, thresholds = precision_recall_curve(df['GT'], df[model_name])

  pr_plot.plot(recall, precision)
  pr_plot.plot([0, 1], [0, 1], '--')
  intrsection_x, intrsection_y = intrsection(recall, precision, [0, 1], [0, 1])
  pr_plot.scatter(intrsection_x, intrsection_y)
  probability_1 = (df['GT']).mean()
  pr_plot.plot([0, 1], [probability_1] * 2, '--')


  fpr, tpr, thresholds = roc_curve(df['GT'], df[model_name])

  roc_plot.plot(fpr, tpr)
  roc_plot.plot([0, 1], [0, 1], '--')
  arg_of_opt = np.argmin(np.abs(tpr + fpr - 1))
  roc_plot.scatter(fpr[arg_of_opt], tpr[arg_of_opt])


  pr_plot.set_ylabel('Precision')
  pr_plot.set_xlabel('Recall')
  pr_plot.set_title('Precision-Recall-крива')
  pr_plot.set_ylim(-0.1, 1.1)
  pr_plot.set_xticks(np.arange(0, 1.1, 0.1))

  roc_plot.set_ylabel('True positive rate')
  roc_plot.set_xlabel('False psitive rate')
  roc_plot.set_title('ROC CURVE')
  roc_plot.set_xticks(np.arange(0, 1.1, 0.1))

  plt.show()

"""## 5."""

np.random.seed(1)

num = 50 + 10 * K

df_1 = df[df['GT'] == 1]
num_delete = int(df_1.shape[0] * num / 100)
indices_to_delete = np.random.choice(df_1.index, size=num_delete, replace=False)
new_df = df.drop(indices_to_delete)

"""## 6."""

print('Частка видалених = ', end='')
print(1 - new_df['GT'].value_counts()[1] / df['GT'].value_counts()[1])
print(new_df['GT'].value_counts())

"""## 7."""

metrics_df, constants_df = get_metrics(new_df, d)

for model_name in ['Model_1', 'Model_2']:
  fig, ax = plt.subplots()
  metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'Balanced Accuracy']
  xs = metrics_df[metrics_df['Model'] == model_name]['Threshold']
  for i in metrics:
    ys = metrics_df[metrics_df['Model'] == model_name][i]
    plt.plot(xs, ys, label=i)
    plt.plot([0, 1], [np.max(ys)]*2, '--')
  metrics = ['AUC for ROC', 'AUC for PRC', 'Youden']
  for i in metrics:
    ys = constants_df[constants_df['Model'] == model_name][i]
    plt.plot([0, 1], [ys]*2, label=i)
  ax.set_xlabel('Threshold')
  ax.set_ylabel('Metric values')
  plt.legend()
  plt.show()

for model_name in ['Model_1', 'Model_2']:
  fig, ax = plt.subplots()
  metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'Balanced Accuracy']
  xs = metrics_df[metrics_df['Model'] == model_name]['Num']
  for i in metrics:
    ys = metrics_df[metrics_df['Model'] == model_name][i]
    plt.plot(xs, ys, label=i)
    plt.plot([0, 1], [np.max(ys)]*2, '--')
    plt.axvline(xs.iloc[np.argmax(ys)], linestyle='--')

  ax.set_xlabel('Num')
  ax.set_ylabel('Metric values')
  plt.legend()
  plt.show()

for model_name in ['Model_1', 'Model_2']:
  figure, (pr_plot, roc_plot) = plt.subplots(ncols = 2, figsize=(15, 5))

  precision, recall, thresholds = precision_recall_curve(new_df['GT'], new_df[model_name])

  pr_plot.plot(recall, precision)
  pr_plot.plot([0, 1], [0, 1], '--')
  intrsection_x, intrsection_y = intrsection(recall, precision, [0, 1], [0, 1])
  pr_plot.scatter(intrsection_x, intrsection_y)
  probability_1 = new_df['GT'].mean()
  pr_plot.plot([0, 1], [probability_1] * 2, '--')


  fpr, tpr, thresholds = roc_curve(new_df['GT'], new_df[model_name])

  roc_plot.plot(fpr, tpr)
  roc_plot.plot([0, 1], [0, 1], '--')
  index = np.argmin(np.abs(tpr + fpr - 1))
  roc_plot.scatter(fpr[index], tpr[index])


  pr_plot.set_ylabel('Precision')
  pr_plot.set_xlabel('Recall')
  pr_plot.set_title('Precision-Recall-крива')
  pr_plot.set_ylim(-0.1, 1.1)
  pr_plot.set_xticks(np.arange(0, 1.1, 0.1))

  roc_plot.set_ylabel('True positive rate')
  roc_plot.set_xlabel('False psitive rate')
  roc_plot.set_title('ROC CURVE')
  roc_plot.set_xticks(np.arange(0, 1.1, 0.1))

  plt.show()

