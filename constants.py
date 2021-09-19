LOG_DIR = "logs"
EDA_DIR = "eda"
RESULT_DIR = "results"
MODEL_DIR = "models"
DATA_DIR = "data"

DATA_FILE = "bank_data.csv"
CHURN_DIST_FILE = "churn_distribution.png"
CUST_AGE_DIST_FILE = "customer_age_distribution.png"
MARITAL_STATUS_DIST_FILE = "martial_status_distribution.png"
TOTAL_TRANS_DIST_FILE = "total_transaction_distribtion.png"
HEATMAP_FILE = "heatmap.png"
FEATURE_IMP_FILE = "feature_importance.png"
FEATURE_SHAP_FILE = "feature_shap.png"
RFC_MODEL_FILE = "rfc_model.pkl"
LRC_MODEL_FILE = "logistic_model.pkl"
RFC_RESULTS_FILE = "rf_results.png"
LRC_RESULTS_FILE = "logistic_results.png"
ROC_RESULTS_FILE = "roc_curve_result.png"

TEST_SIZE = 0.3

LOG_FILE = "churn_library.log"

RESPONSE = "Churn"

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

KEEP_COLS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_' + RESPONSE, 'Education_Level_' + RESPONSE, 'Marital_Status_' + RESPONSE, 
             'Income_Category_' + RESPONSE, 'Card_Category_' + RESPONSE]

