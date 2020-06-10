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
from aif360.explainers import MetricTextExplainer

#Markdown 
from IPython.display import Markdown, display


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
    binary_dataset = BinaryLabelDataset(favorable_label=1, 
                                        unfavorable_label=0, 
                                        df=df_data[0], 
                                        label_names=['salary'],
                  protected_attribute_names=['sex'])
    
    return binary_dataset

def create_multiple_datasets(dataset_creator, salary_thresh_list):
    binary_dataset_list = []
    #Create Datasets with Different Salaries 
    for salary in salary_thresh_list:
        binary_dataset = dataset_creator(salary)
        binary_dataset_list.append(binary_dataset)
    return binary_dataset_list


def create_binary_dataset_sb():    
    """This will create a binary dataset from the csv with a set salary 
    as the threshold for later predictions.
    
    Input - A numeric salary to be set as the threshold
    
    Out - A AIF360 binary dataset with one-hot encoded categorical columns
    """
    
    data = pd.read_csv('../company_x_sb.csv', index_col='employee_id')
    data_with_label = data.copy()
    data_with_label['sex'] = data_with_label['sex'].transform(lambda x: x == 'M').astype(int)

    std_data = StandardDataset(df=data_with_label,   
                             label_name='new_signing_bonus',
                             favorable_classes =[1],
                            protected_attribute_names=['sex'], 
                             privileged_classes=[[1]],
                            categorical_features=['degree_level', 'dept'], 
                              features_to_drop=['boss_id'])

    df_data = std_data.convert_to_dataframe()
    binary_dataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=df_data[0], label_names=['new_signing_bonus'],
                  protected_attribute_names=['sex'])
    
    return binary_dataset


def show_metrics(binary_dataset_list):
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


def get_bias_amplification(train_data, prediction_data):
    privileged_groups= [{'sex':1}]
    unprivileged_groups= [{'sex': 0}]
    train_metrics = BinaryLabelDatasetMetric(train_data, 
                            unprivileged_groups=unprivileged_groups, 
                            privileged_groups=privileged_groups)
    
    prediction_metrics = BinaryLabelDatasetMetric(prediction_data, 
                            unprivileged_groups=unprivileged_groups, 
                            privileged_groups=privileged_groups)
    
    
    tedf = train_metrics.smoothed_empirical_differential_fairness()
    pedf = prediction_metrics.smoothed_empirical_differential_fairness()
    bias_amp = pedf - tedf
    return bias_amp


def plot_mean_diff_and_di(mean_list_1, di_list_1, mean_list_2, di_list_2, path):
    fig, axes = plt.subplots(2, 1, figsize = (8,6))
    x_axis = range(len(mean_list_1))

    axes[0].bar(x_axis, mean_list_1, label='Mean Difference' )
    axes[0].bar(x_axis, di_list_1, label='1 - Disparate Impact')
    axes[0].axhline(0)
    axes[0].set_xticks(x_axis)
    axes[0].set_xticklabels([150000,200000])
    axes[0].set_title('Metrics at Different Salary Threshold')
    axes[0].set_xlabel('Salary Threshhold for Classifier Split')
    axes[0].set_ylim(-.5, .5)

    axes[1].bar(x_axis, mean_list_2, label='Mean Difference' )
    axes[1].bar(x_axis, di_list_2, label='1 - Disparate Impact')
    axes[1].axhline(0)
    axes[1].set_xticks(x_axis)
    axes[1].set_xticklabels([150000,200000])
    axes[1].set_title('Metrics at Different Salary Threshold')
    axes[1].set_xlabel('Salary Threshhold for Classifier Split')
    axes[1].set_ylim(-.5, .5)

    plt.tight_layout()
    axes[0].legend()
    axes[1].legend()
    plt.savefig(path)