import lime
from lime.lime_tabular import LimeTabularExplainer
from aif360.datasets.lime_encoder import LimeEncoder
import pickle

def get_lime_model(train, test):
    lime_list = []
    lime_data = LimeEncoder().fit(train)

    s_train = lime_data.transform(train.features)
    s_test = lime_data.transform(test.features)

    explainer = LimeTabularExplainer(
        s_train, class_names=lime_data.s_class_names, 
        feature_names=lime_data.s_feature_names,
        categorical_features=lime_data.s_categorical_features, 
        categorical_names=lime_data.s_categorical_names, 
        kernel_width=3, verbose=False, discretize_continuous=True)
    lime_list.append(lime_data)
    lime_list.append(s_test)
    lime_list.append(explainer)
    return lime_list



def s_predict_fn(model, x):
    return model.predict_proba(lime_list[0].inverse_transform(x))

def show_explanation_html(ind):
    exp = lime_list[2].explain_instance(lime_list[0][ind], s_predict_fn, num_features=10)
    html = exp.as_html(show_all=False)
    return html 