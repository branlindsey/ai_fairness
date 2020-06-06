#Standard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#AIF360 Dataset Classes 
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset

#AIF360 Metrics Classes  
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import ClassificationMetric



def create_binary_dataset_salary(salary):    
    """This will create a binary dataset from the csv with a set salary 
    as the threshold for later predictions.
    
    Input - A numeric salary to be set as the threshold
    
    Out - A AIF360 binary dataset with one-hot encoded categorical columns
    """
    
    data = pd.read_csv('../company_x.csv', index_col='employee_id')
    data_with_label = data.copy()
    data_with_label['salary'] = data_with_label['salary'].transform(lambda x: x > salary).astype(int)
    data_with_label['sex'] = data_with_label['sex'].transform(lambda x: x == 'M').astype(int)

    std_data = StandardDataset(df=data_with_label,   
                             label_name='salary',
                             favorable_classes =[1],
                            protected_attribute_names=['sex'], 
                             privileged_classes=[[1]],
                            categorical_features=['degree_level', 'dept'], 
                              features_to_drop=['boss_id'])

    df_data = std_data.convert_to_dataframe()
    binary_dataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=df_data[0], label_names=['salary'],
                  protected_attribute_names=['sex'])
    
    return binary_dataset

def create_multiple_datasets(dataset_creator, salary_thresh_list):
    binary_dataset_list = []
    #Create Datasets with Different Salaries 
    for salary in salary_thresh_list:
        binary_dataset = dataset_creator(salary)
        binary_dataset_list.append(binary_dataset)
    return binary_dataset_list


def get_dataset_metrics(binary_dataset_list):
    #Set privileged and unprivileged groups
    privileged_groups= [{'sex':1}]
    unprivileged_groups= [{'sex': 0}]
    
    for dataset in binary_dataset_list:
        display(Markdown("#### Model  dataset metrics"))
        metrics = BinaryLabelDatasetMetric(dataset, 
                            unprivileged_groups=unprivileged_groups, 
                            privileged_groups=privileged_groups)

        ex_metrics = MetricTextExplainer(metrics)

        print(ex_metrics.mean_difference())
        print('\n')
        print(ex_metrics.disparate_impact())


def get_dataset_metrics_list(binary_dataset_list):
    #Set privileged and unprivileged groups
    privileged_groups= [{'sex':1}]
    unprivileged_groups= [{'sex': 0}]
    
    mean_diff_list = []
    disp_imp_list = []
    for dataset in binary_dataset_list:
        metrics = BinaryLabelDatasetMetric(dataset, 
                            unprivileged_groups=unprivileged_groups, 
                            privileged_groups=privileged_groups)
        mean_diff_list.append(metrics.mean_difference())
        disp_imp_list.append(1 - metrics.disparate_impact())
    return mean_diff_list, disp_imp_list


