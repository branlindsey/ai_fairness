from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.metrics import ClassificationMetric


def create_split_lists(dataset_list):
    #Create Train, Validation, Test Sets
    data_train_list =[]
    data_val_list = []
    data_test_list =[]
    for dataset in dataset_list:
        data_train_val, data_test = dataset.split([0.8], shuffle=True)
        data_train, data_val = data_train_val.split([0.8], shuffle=True)

        data_train_list.append(data_train)
        data_val_list.append(data_val) 
        data_test_list.append(data_test) 
    return data_train_list, data_val_list, data_test_list 

def fit_models(estimator, train_list):
    fit_list = []
    for train_ in train_list:
        X_train = train_.features
        y_train = train_.labels.ravel()
        
        model = estimator
        model.fit(X_train, y_train)
        fit_list.append(model)
    return fit_list

def get_predictions(estimator, train_list, test_list):
    pred_list = []
    for train_, test_ in zip(train_list, test_list):
        X_train = train_.features
        y_train = train_.labels.ravel()
        X_test = test_.features  
        model = estimator
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds = test_.copy()
        preds.labels = y_pred.ravel()
        pred_list.append(preds)
    return pred_list

def get_rw_predictions(estimator, rw_train_list, rw_test_list):
    rw_pred_list = []
    for train_, test_ in zip(rw_train_list, rw_test_list):
        X_train = train_.features
        y_train = train_.labels.ravel()
        X_test = test_.features  
        model = estimator
        model.fit(X_train, y_train, sample_weight=train_.instance_weights)
        y_pred = model.predict(X_test)
        preds = test_.copy()
        preds.labels = y_pred.ravel()
        rw_pred_list.append(preds)
    return rw_pred_list



    

def show_classifier_metrics(test_list, prediction_list):
    privileged_groups= [{'sex':1}]
    unprivileged_groups= [{'sex': 0}]
    
    
    counter = 1 
    for test_, pred_ in zip(test_list, prediction_list):
        
        display(Markdown("#### Model {}  dataset metrics".format(counter)))
        

        model_metric = ClassificationMetric(test_, pred_, 
                            unprivileged_groups=unprivileged_groups, 
                            privileged_groups=privileged_groups)

        ex_model_metric= MetricTextExplainer(model_metric)
        print(ex_model_metric.average_odds_difference())

        print('Difference in Recall between Unprivileged and Privileged: {:.3f}'
          .format(model_metric.equal_opportunity_difference()))

        print('Difference in Precision between Unprivileged and Privileged: {:.3f}.'
            .format(model_metric.precision(privileged=False)- model_metric.precision(privileged=True)))
        counter +=1
        

def get_classifier_metrics(test_list, prediction_list):
    privileged_groups= [{'sex':1}]
    unprivileged_groups= [{'sex': 0}]
    acc_list = []
    bal_acc_list = []
    avg_odds_list=[]
    recall_diff_list = []
    precision_diff_list = []
    for test_, pred_ in zip(test_list, prediction_list):
        model_metric = ClassificationMetric(test_, pred_, 
                            unprivileged_groups=unprivileged_groups, 
                            privileged_groups=privileged_groups)
        
        acc_list.append(model_metric.accuracy().round(3))
        bal_acc_list.append(((model_metric.true_positive_rate()+ model_metric.true_negative_rate())/2).round(3))
        avg_odds_list.append(model_metric.average_odds_difference().round(3))
        recall_diff_list.append(model_metric.equal_opportunity_difference().round(3))
        precision_diff_list.append((model_metric.precision(privileged=False)- model_metric.precision(privileged=True)).round(3))  
    return acc_list, bal_acc_list, avg_odds_list, recall_diff_list, precision_diff_list


def get_confusion_matrix(test_list, prediction_list):
    privileged_groups= [{'sex':1}]
    unprivileged_groups= [{'sex': 0}]
   
    model_metric = ClassificationMetric(test_list, prediction_list, 
                        unprivileged_groups=unprivileged_groups, 
                        privileged_groups=privileged_groups)
    
    priv_conf_mat = model_metric.binary_confusion_matrix(privileged=True)
    unpriv_conf_mat = model_metric.binary_confusion_matrix(privileged=False)
    return priv_conf_mat, unpriv_conf_mat