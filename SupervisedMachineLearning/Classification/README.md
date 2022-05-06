Evaluation metrics in classification model:

Precision and Recall:
<img width="1025" alt="image" src="https://user-images.githubusercontent.com/31846843/167158921-8a085a81-7cd5-40bf-946e-5b2424e56ca5.png">

Precision is important metric 
How do you change a model if it is producing too many false positive results?
Increase the threshold

Recall is quite useful in scenarios where we are trying to identify the fradulent transaction out of many transactions.
How do you change a model if it is producing too many false negative results?
Decrease the threshold

F1 score: Identifies a balance between precision and recall. The F1-score combines the precision and recall of a classifier into a single metric 
by taking their harmonic mean. It is primarily used to compare the performance of two classifiers. Suppose that classifier A has a higher recall, and classifier B has higher precision.
F1 score = 2 precision * recall / (precision + recall)


ROC Curve:
An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. The following figure shows a typical ROC curve.
AUC stands for "Area under the ROC Curve." That is, AUC measures the entire two-dimensional area underneath the entire ROC curve
<img width="344" alt="image" src="https://user-images.githubusercontent.com/31846843/167161169-e542bae2-cf5c-45e8-82b4-50b73d593e44.png">
AUC is desirable for the following two reasons:

AUC is independent of scale. It measures the quality of the model's predictions at the given threshold no matter what it is.

Residuals:
<img width="772" alt="image" src="https://user-images.githubusercontent.com/31846843/167158586-9f308cb3-f97d-4713-ba03-d5471a261102.png">
