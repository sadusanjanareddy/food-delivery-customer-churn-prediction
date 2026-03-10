import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
accuracy_score, precision_score, recall_score,f1_score,roc_auc_score, roc_curve, classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("online_delivery_data.csv")
#--- Make Output numeric (0/1) ---
print("columns names:")
print(df.columns)
# --- Clean and Convert Output Column ---
temp_out = df["Output"].astype(str).str.strip().str.lower()

mapping = {
    "1": 1, "0": 0,
    "yes": 1, "no": 0,
    "churn": 1, "not churn": 0,
    "churned": 1, "not churned": 0,
    "leave": 1, "stay": 0,
    "left": 1, "retained": 0,
    "loyal": 0, "not loyal": 1
}

df["Output"] = temp_out.map(mapping)

# Check if any value was not mapped
if df["Output"].isna().sum() > 0:
    print("Warning: Some Output values were not mapped!")
    print(df["Output"].isna().sum())
df["Output"] = df["Output"].astype(int)

print("Final Output distribution:")
print(df["Output"].value_counts())
print("Output dtype:", df["Output"].dtype)
    #try mapping first
df["Output"]=temp_out.map(mapping)
    # if still NaN, fallback:factorize (first class=0, second class=1)
if df["Output"].isna().any():
        df ["Output"] = pd.factorize(temp_out)[0]
        df["Output"]=pd.to_numeric(df["Output"],errors="coerce").fillna(0).astype(int)
        
print("Unique values in Output column:")
print(df["Output"].value_counts())
print("\n================================")
print("DATASET SHAPE:",df.shape)
print ("===============================")
print("First 5 rows:")
print (df.head())
print ("\nColumn names:")
print(list(df.columns))
#basic checks
if "Output" not in df.columns:
    raise ValueError ("Target column 'Output' not found in the pdataset. Please check the Csv file.")
# EDA
# Missing Value Analysis
print("\n======================================================")
print("MISSING VALUE ANALYSIS")
print("========================================================")
missing_counts = df.isna().sum().sort_values (ascending=False)
missing_percent = (df.isna().mean()*100).sort_values(ascending=False)
missing_table = pd.DataFrame({"missing_count": missing_counts, "missing_percent": missing_percent})
print(missing_table[missing_table["missing_count"]>0])

plt.figure()
(missing_percent[missing_percent > 0]).plot(kind="bar")
plt.title("Missing values % by column (only colums with missing values)")
plt.ylabel("Missing %")
plt.tight_layout()
plt.show()
# Univariate Analysis - Distribution of age, income, family size
print ("\n==============================================================")
print("UNIVARIATE ANALYSIS (Age, Monthiy Income, Family size)")
print("=================================================================")

# Age distribution
if "Age" in df.columns:
    plt.figure()
    df["Age"].dropna().hist(bins=20)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
else:
    print ("Column 'Age' not found. Skipping Age distribution.")
# Monthly Income distribution (categorical)if "Monthly Income”in df.columns:
if "Monthly Income" in df.columns:
    plt.figure()
    df ["Monthly Income"].astype (str).value_counts().plot(kind="bar")
    plt.title("Monthly 1 Income Distribution")
    plt.xlabel("Monthly  Income")
    plt.ylabel("Count")
    plt.xticks(rotation=45,ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("Column 'Monthly Income' not found. Skipping Income distribution.")
# Family size distribution
if "Family size" in df.columns:
    plt.figure()
    df["Family size"].dropna().hist(bins=15)
    plt.title("Family Size Distribution")
    plt.xlabel("Family size")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
else:
    print ("Column 'Family size' not found. Skipping Family size distribution.")
# Categorical Analysis Churn rate by gender, income,and occupation
print("\n==================================================================")
print("CATEGORICAL ANALYSIS (Churn rate by Gender, Monthly Income, occupation)")
print("===================================================================")
def churn_rate_bar(category_col):
    if category_col not in df.columns:
        print(f"column'icategory_coll' not found. Skipping.")
        return
    temp = df[[category_col, "Output"]].copy()
    temp[category_col] = temp[category_col].astype(str)
# Churn rate - mean (Output) if output is 0/1
    rate =temp.groupby(category_col)["Output"].mean().sort_values (ascending=False)
    plt.figure()
    rate.plot(kind="bar")
    plt.title(f"Churn Rate by {category_col}")
    plt.ylabel ("Churn Rate (mean of Output)")
    plt.xlabel(category_col)
    plt.xticks(rotation=45,ha="right")
    plt.tight_layout()
    plt.show()
churn_rate_bar("Gender")
churn_rate_bar("Monthly Income")
churn_rate_bar("Occupation")
# Correlation Analysis - Relationship between delivery issues and churn
# Note: We will create a numeric-only dataframe (after encoding), and show correlations with Output.
print("\n======================================")
print ("CORRELATION ANALYSIS")
print ("=========================================")
# we'll do a quick numeric conversion ONLY for correlation viewing:
temp_for_corr = df.copy()
# Identify columns
target = "Output"
feature_cols = [c for c in temp_for_corr.columns if c != target]
# Convert object columns to dunnies for correlation visualization 
temp_encoded = pd.get_dummies(temp_for_corr[feature_cols],drop_first=True)
temp_encoded[target] = temp_for_corr[target].astype(float)
corr = temp_encoded.corr(numeric_only=True)
# correlation with target
corr_with_target = corr[target].sort_values(ascending=False)
print ("\nTop positive correlations with output:")
print (corr_with_target.head(15))
print("\nrop negative correlations with Output:")
print(corr_with_target.tail(15))

plt.figure(figsize=(10, 6))
corr_with_target.drop(target).head(25).plot(kind="bar")
plt.title("Top correlations with Output lencoded features)")
plt.xlabel("Correlation")
plt.tight_layout()
plt.show()
# Churn Distribution - class balance visualization
print("\n====================================================")
print ("CHURN DISTRIBOTION")
print ("=====================================================")
print(df["Output"].value_counts())
plt.figure()
df["Output"].value_counts().plot(kind="bar")
plt.title("Churn Distribution (Output)")
plt.xlabel("Output")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
# Service Issue Impact - Late delivery, wrong order vs churn
# (These columns exist in many versions of this dataset; we check before using.)
print("\n==================================================")
print("SERVICE ISSOE IMPACT (Late Delivery, Wrong order delivered)")
print ("===================================================")
service_cols = ["Late Delivery", "wrong order delivered"]
for col in service_cols:
    if col in df.columns:
        temp = df[[col, "Output"]].copy()
        # convert to string for grouping,in case values are labels
        temp[col]= temp[col].astype(str)
        rate =temp.groupby(col) ["Output"].mean().sort_values(ascending=False)
        plt.figure()
        rate.plot (kind="bar")
        plt.title(f"Churn Rate by {col}")
        plt.ylabel("Churn Rate")
        plt.xlabel (col)
        plt.xticks(rotation=45,ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Column 'Icol)' not found. Skipping.")
#--------
# Preprocessing
print ("\n====================================================")
print ("PREPROCESSING")
print ("======================================================")
# Separate x and y
X = df.drop(columns=["Output"]).copy()
y = df["Output"].copy()
# Identify col unn types
numeric_cols =X.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]
print("Numeric columns:",len(numeric_cols))
print("Categorical columns:", len(categorical_cols))
# Handle missing values
#-numeric:fill with median 
# #- categorical: fill with mode 
for col in numeric_cols:
    if X[col].isna().sum()>0:
        X[col]=X[col].fillna(X[col].median())
for col in categorical_cols:
    if X[col].isna().sum()>0:
        X[col]=X[col].fillna(X[col].mode()[0])
# One-hot encode categorical features
X_encoded = pd.get_dummies(X,columns=categorical_cols,drop_first=True)
print ("\nShape after encoding:", X_encoded.shape)
# Train-test split
X_train, X_test, y_train, y_test= train_test_split(X_encoded, y, test_size=0.2,random_state=42, stratify=y)

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)
# Train models
print ("\n=====================================")
print("MODEL TRAINING")
models = {}
# Logistic Regression
models["Logistic Regression"]=LogisticRegression(max_iter=2000)
# Random Forest
models["Random Forest"] = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
# XGBoost
models["XGBoost"] = XGBClassifier(n_estimators=400,max_depth=5,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,random_state=42,eval_metric="logloss")
results = []
for name in models:
    model=models[name]
    model.fit(X_train, y_train)
# predictons
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob=model.predict_proba(X_test) [:,1]
    else:
 #fallback for models without predict_proba

        y_prob = None
acc = accuracy_score(y_test, y_pred)
prec= precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test,y_pred,zero_division=0)

if y_prob is not None:
    roc_auc = roc_auc_score(y_test, y_prob)
else:
    roc_auc = np.nan
results.append([name, acc, prec, rec, f1, roc_auc])
print("\n-----------------------------------------------")
print("Model:",name)
print("Accuracy:", round(acc, 4))
print("precision:",round(prec,4))
print("Recall:", round(rec, 4))
print("F1:", round(f1, 4))
print ("ROC-AUC:", round(roc_auc, 4) if not np.isnan(roc_auc) else "NA.")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report:")
print (classification_report(y_test, y_pred, zero_division=0))
# Comparison table
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision","recall","F1", "ROC_AUC"]).sort_values (by="ROC_AUC",ascending=False)
print("\n================================")
print ("MODEL COMPARISON (Sorted by ROC_AOC)")
print(results_df)
plt.figure()
plt.bar(results_df["Model"], results_df["ROC_AUC"])
plt.title("Model Comparison by ROC-AUC")
plt.ylabel("ROC-AUC")
plt.xticks(rotation=30,ha="right")
plt.tight_layout()
plt.show()
# ROC curve for best model
best_model_name = results_df.iloc[0]["Model"]
best_model=models[best_model_name]
print("\n===================================")
print("BEST MODEL:",best_model_name)
print("======================================")
best_prob = best_model.predict_proba(X_test) [:,1] 
fpr , tpr, thresholds = roc_curve(y_test, best_prob)
best_auc = roc_auc_score(y_test, best_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"{best_model_name} (AUC={best_auc:.3f})")
plt.plot([0, 1],[0, 1], linestyle="--")
plt.title("ROC Curve (Best Model)")
plt.xlabel("False Positive Rate")
plt.ylabel ("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()
# Predict churn for a new customer(example)
print("\n=====================================")
print("SINGLE CUSTOMER PREDICTION")
print("======================================")
new_customer = {}
for col in X.columns:
    new_customer[col]= X[col].iloc[0] #default:take frst row as safe sample
# Example: you can override few fields (if they exist)
if "Age" in new_customer:
    new_customer["Age"] = 28
if "Gender" in new_customer:
    new_customer ["Gender"] = "Male"
# Convert to DataFrame
new_df =pd.DataFrame([new_customer])
# Missing handling (same logic as training)
for col in numeric_cols:
    if col in new_df.columns:
        new_df [col] = pd.to_numeric(new_df[col], errors="coerce")
        new_df [col]=new_df[col].fillna(X[col].median())
for col in categorical_cols:
    if col in new_df.columns:
        new_df [col] = new_df[col].astype(str)
        new_df[col] =new_df[col].fillna(X[col].mode()[0])
# One-hot encode
new_encoded = pd.get_dummies(new_df, columns=categorical_cols, 
drop_first=True)
#Align columns with training data
#Any missing columns in new_encoded should be created with 0
for col in X_encoded.columns:
    if col not in new_encoded.columns:
        new_encoded [col] =0
# Extra columns (rare) should be dropped
new_encoded = new_encoded [X_encoded.columns]
# Predict
pred_label = best_model.predict(new_encoded) [0]
pred_prob = best_model.predict_proba(new_encoded) [0, 1]
if pred_label == 1:
    print(f"Prediction:CUSTOMER WILL CHURN(probability={pred_prob:.3f})")
else:
    print(f"Prediction: CUSTOMER WILL NOT CHURN(probability={pred_prob:.3f})")