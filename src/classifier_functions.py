
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def roc_curve_(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    sorted_ = np.sort(probabilities)
    threshold = list(sorted_)
    true_pos = labels.sum()
    true_neg =  len(labels)-true_pos
    
    TPR =[]
    FPR = []
    for vals in threshold:        
        pos = np.sum((vals <= probabilities) & (labels==True))
        f_pos = np.sum((vals <= probabilities) & (labels==False))

        TPR.append(pos/true_pos)
        FPR.append(f_pos/true_neg)
    return TPR, FPR, threshold

def standard_confusion_matrix(y_true, y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    tp = np.sum((y_true == y_predict) & (y_predict==1))
    fp = np.sum((y_true != y_predict) & (y_predict==1))
    fn = np.sum((y_true != y_predict) & (y_predict==0))
    tn = np.sum((y_true == y_predict) & (y_predict==0))
   
    return np.array([[tp, fp], [fn, tn]])


def confusion_matrix_to_df(confusion_matrix):
    df = pd.DataFrame(confusion_matrix, columns=['Actual True', 'Actual False'], 
                      index=['Predicted True', 'Predicted False'])
    return df                                                       

def confusion_matrix_to_df_from_dict(confusion_matrix):
    matrix = np.zeros([2,2])
    conf_list = list(confusion_matrix.values())
    matrix[0,0] = conf_list[0].round()
    matrix[0,1] = conf_list[1].round()
    matrix[1,0] = conf_list[3].round()
    matrix[1,1] = conf_list[2].round()
    
    df = pd.DataFrame(matrix, columns=['Actual True', 'Actual False'], 
                      index=['Predicted True', 'Predicted False'])
    return df                

def scorecard(y_true, y_pred):
    print(f'The Accuracy score is {accuracy_score(y_true, y_pred):2.3f}.\n')
    print(f'The Precision score is {precision_score(y_true, y_pred):2.3f}.\n')
    print(f'The Recall score is {recall_score(y_true, y_pred):2.3f}.\n')
    print(f'      Confusion Matrix')
    return confusion_matrix_to_df(standard_confusion_matrix(y_true, y_pred))
                                                                      