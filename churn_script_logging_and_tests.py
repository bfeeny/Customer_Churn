'''
Churn Analysis logging and tests

author: Brian Feeny <bfeeny@mac.com>
date: September 17, 2021
'''
import os
import logging
import numpy as np
import pandas as pd
import churn_library as cls
import constants as const

logging.basicConfig(
    filename=os.path.join(const.LOG_DIR, const.LOG_FILE),
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import_data():
    '''
    test data import
    '''
    try:
        df = cls.import_data(os.path.join(const.DATA_DIR, const.DATA_FILE))
        logging.info("SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("FAIL: Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_add_response_column():
    '''
    test add response column
    '''
    df = cls.import_data(os.path.join(const.DATA_DIR, const.DATA_FILE))
    try:
        df = cls.add_response_column(df)
        assert const.RESPONSE in df.columns
        logging.info('SUCCESS: Testing add_response_column')
    except AssertionError as err:
        logging.error(
            'Testing add_response_column: There is not a column called %s',
            const.RESPONSE)
        raise err


def test_perform_eda():
    """
    test perform_eda
    """
    try:
        assert os.path.exists(const.EDA_DIR)
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The directory %s does not exist.",
            const.EDA_DIR)
        raise err
    df = cls.import_data(os.path.join(const.DATA_DIR, const.DATA_FILE))
    df = cls.add_response_column(df)
    try:
        cls.perform_eda(df)
        assert os.path.exists(
            os.path.join(
                const.EDA_DIR,
                const.CHURN_DIST_FILE))
        assert os.path.exists(
            os.path.join(
                const.EDA_DIR,
                const.CUST_AGE_DIST_FILE))
        assert os.path.exists(
            os.path.join(
                const.EDA_DIR,
                const.MARITAL_STATUS_DIST_FILE))
        assert os.path.exists(
            os.path.join(
                const.EDA_DIR,
                const.TOTAL_TRANS_DIST_FILE))
        assert os.path.exists(os.path.join(const.EDA_DIR, const.HEATMAP_FILE))
        logging.info('SUCCESS: Testing perform_eda')
    except AssertionError as err:
        logging.error('Testing perform_eda: A file is missing.')
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    df = cls.import_data(os.path.join(const.DATA_DIR, const.DATA_FILE))
    df = cls.add_response_column(df)
    try:
        df = cls.encoder_helper(df, const.CAT_COLUMNS, const.RESPONSE)
        assert set(name + '_' + const.RESPONSE for name in const.CAT_COLUMNS
                   ).issubset(df.columns)
        logging.info('SUCCESS: Testing encoder_helper')
    except KeyError as err:
        logging.error(
            'Testing encoder_helper: An encoded categorical column is missing')
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df = cls.import_data(os.path.join(const.DATA_DIR, const.DATA_FILE))
    df = cls.add_response_column(df)
    df = cls.encoder_helper(df, const.CAT_COLUMNS, const.RESPONSE)
    try:
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df, const.RESPONSE)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        logging.info(
            'SUCCESS: Testing perform_feature_engineering')
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering: not all features found!')
        raise err
    try:
        train_size = int(np.floor(df.shape[0] * (1 - const.TEST_SIZE)))
        test_size = df.shape[0] - train_size
        assert X_train.shape[0] == train_size
        assert X_train.shape[1] == len(const.KEEP_COLS)
        assert X_test.shape[0] == test_size
        assert X_test.shape[1] == len(const.KEEP_COLS)
        assert y_train.shape[0] == train_size
        assert y_test.shape[0] == test_size
        logging.info(
            'SUCCESS: Testing perform_feature_engineering: all shapes are correct')
    except AssertionError as err:
        logging.error('Testing perform_feature_engineering: incorrect dimensions for features.')
        logging.error('Expected X_train(%s,%s), X_test(%s,%s), y_train(%s,), y_test(%s,).'
                      ,train_size, len(const.KEEP_COLS), test_size, len(const.KEEP_COLS),
                       train_size, test_size)
        logging.error('Got X_train(%s,%s), X_test(%s,%s), y_train(%s,), y_test(%s,).',
                      X_train.shape[0],X_train.shape[1], X_test.shape[0], X_test.shape[1],
                      y_train.shape[0], y_test.shape[0])
        raise err


def test_train_models():
    '''
    test train_models
    '''
    try:
        assert os.path.exists(
            os.path.join(
                const.MODEL_DIR,
                const.LRC_MODEL_FILE))
        assert os.path.exists(
            os.path.join(
                const.MODEL_DIR,
                const.RFC_MODEL_FILE))
        logging.info('SUCCESS: Testing train_model')
    except AssertionError as err:
        logging.error('Testing train_model: model file missing.')
        raise err


if __name__ == "__main__":
    test_import_data()
    test_add_response_column()
    test_perform_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
