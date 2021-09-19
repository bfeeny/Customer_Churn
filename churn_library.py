# library doc string
'''
Churn Analysis

author: Brian Feeny <bfeeny@mac.com>
date: September 17, 2021
'''

# import libraries
import logging
import os
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import constants as const

sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename=os.path.join(const.LOG_DIR, const.LOG_FILE),
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        logging.info("SUCCESS: %s loaded shape %s", pth, df.shape)
    except FileNotFoundError:
        logging.error("ERROR: %s not found.", pth)
    return df

def add_response_column(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            df: pandas dataframe with response column added
    '''
    df[const.RESPONSE] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return df

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    df[const.RESPONSE].hist().get_figure()
    plt.savefig(os.path.join(const.EDA_DIR, const.CHURN_DIST_FILE))

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(os.path.join(const.EDA_DIR, const.CUST_AGE_DIST_FILE))

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join(const.EDA_DIR, const.MARITAL_STATUS_DIST_FILE))

    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.savefig(os.path.join(const.EDA_DIR, const.TOTAL_TRANS_DIST_FILE))

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(const.EDA_DIR, const.HEATMAP_FILE))


def encoder_helper(df, category_lst, response=const.RESPONSE):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used \
            for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        cat_groups = df.groupby(category).mean()[response]
        cat_lst = [cat_groups.loc[val] for val in df[category]]
        df['_'.join([category, response])] = cat_lst

    return df


def perform_feature_engineering(df, response=const.RESPONSE):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be \
              used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = df[const.KEEP_COLS]
    y = df[response]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=const.TEST_SIZE, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    fig = plt.figure(figsize=(6, 5))
    fig.add_subplot(1, 1, 1)
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(os.path.join(const.RESULT_DIR, const.RFC_RESULTS_FILE))

    fig = plt.figure(figsize=(6, 5))
    fig.add_subplot(1, 1, 1)
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(os.path.join(const.RESULT_DIR, const.LRC_RESULTS_FILE))

    # random forest results
    print("random forest results")
    print("test results")
    print(classification_report(y_test, y_test_preds_rf))
    print("train results")
    print(classification_report(y_train, y_train_preds_rf))

    # logistic results
    print("logistic regression results")
    print("test results")
    print(classification_report(y_test, y_test_preds_lr))
    print("train results")
    print(classification_report(y_train, y_train_preds_lr))


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # save graph
    plt.savefig(os.path.join(output_pth, const.FEATURE_IMP_FILE))

    # shap plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    plt.figure(figsize=(15, 8))
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.savefig(os.path.join(output_pth, const.FEATURE_SHAP_FILE))


def roc_curve_plot(rfc, lrc, X_test, y_test, output_pth):
    '''
    creates and stores the roc plot in pth
    input:
            rfc: random forest model
            lrc: logistic regession model
            X_test: test data covariate matrix
            y_test: test data ground truths
            output_pth: path to store the figure

    output:
             None
    '''
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(os.path.join(output_pth, const.ROC_RESULTS_FILE))


def get_preds(X_train, X_test, lrc, rfc):
    '''
    get predictions from models
    input:
            X_train: model training data
            X_test: model test data
            lrc: logistic regression model
            rfc: random forest model

    output:
             y_train_preds_lr: predictions from the logistic model train data
             y_train_preds_rf: predictions from the random forest model train data
             y_test_preds_lr: predictions from the logistic model test data
             y_test_preds_rf: predictions from the random forest model test data
    '''
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf


def load_models():
    '''
    load a previously saved model

    output:
              logistic regression model and random forest model
    '''
    lrc = joblib.load(os.path.join(const.MODEL_DIR, const.LRC_MODEL_FILE))
    rfc = joblib.load(os.path.join(const.MODEL_DIR, const.RFC_MODEL_FILE))

    return lrc, rfc


def train_models(X_train, y_train):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              y_train: y training data
    output:
              None
    '''
    # logistic regression
    lrc = LogisticRegression(max_iter=1000)
    lrc.fit(X_train, y_train)

    # random forest
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # save best model
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            const.MODEL_DIR,
            const.RFC_MODEL_FILE))
    joblib.dump(lrc, os.path.join(const.MODEL_DIR, const.LRC_MODEL_FILE))


if __name__ == "__main__":
    data = import_data(os.path.join(const.DATA_DIR, const.DATA_FILE))
    print("data 1: ", data.shape)
    data = add_response_column(data)
    print("data 2: ", data.shape)
    perform_eda(data)
    data = encoder_helper(data, const.CAT_COLUMNS, const.RESPONSE)
    xTr, xTe, yTr, yTe = perform_feature_engineering(data, const.RESPONSE)
    train_models(xTr, yTr)
    lrc_model, rfc_model = load_models()
    yTr_pred_lr, yTr_pred_rf, yTe_pred_lr, yTe_pred_rf = get_preds(
        xTr, xTe, lrc_model, rfc_model)
    feature_importance_plot(rfc_model, xTr, const.RESULT_DIR)
    classification_report_image(
        yTr,
        yTe,
        yTr_pred_lr,
        yTr_pred_rf,
        yTe_pred_lr,
        yTe_pred_rf)
    roc_curve_plot(rfc_model, lrc_model, xTe, yTe, const.RESULT_DIR)
