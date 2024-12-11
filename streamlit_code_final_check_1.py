import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from imblearn.under_sampling import RandomUnderSampler
from collections import OrderedDict
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense, Dropout
import warnings
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import label_binarize

# Suppress warnings
warnings.filterwarnings("ignore")

def show_problem_statement():
    """Display the problem statement for the project."""
    st.title("Project Problem Statement")
    st.write("""Problem Statement Story: Lending Risk Analysis
In a bustling city, a lending institution named ProsperFinance was thriving, helping individuals achieve their dreams by providing loans for homes, education, and businesses. However, as the business grew, ProsperFinance faced a daunting challenge: how to effectively identify borrowers who might pose a higher risk of defaulting on their loans. Each default not only impacted the company financially but also its ability to extend opportunities to other borrowers.

The company’s decision-makers realized they were sitting on a treasure trove of data. They had records of previous applicants, including their age, income, homeownership status, loan amount requested, credit history, payment behavior, and past loan performance. However, manually sifting through this data was inefficient, subjective, and prone to error.

Recognizing the need for a robust solution, ProsperFinance sought to develop a cutting-edge web application to automate this process. The app needed to analyze the data, identify patterns, and predict the risk associated with new loan applicants. The ultimate goal was twofold: to safeguard the company’s resources and maintain fairness in lending by basing decisions on clear, data-driven insights.

To achieve this, a team of data scientists began by conducting thorough exploratory data analysis (EDA) to uncover key factors influencing loan defaults. They then engineered features like debt-to-income ratios and payment consistency to enhance predictive accuracy. Leveraging advanced algorithms such as Gradient Boosting, Support Vector Machines (SVM), Artificial Neural Networks (ANN), and Optimally Weighted Fuzzy K-Nearest Neighbors (OWFKNN), they designed a comprehensive risk evaluation system.

The project culminated in a user-friendly Streamlit dashboard. This app not only provided intuitive risk predictions but also offered a behind-the-scenes view of the data analysis and modeling process for stakeholders, ensuring trust and transparency. By integrating oversampling techniques like SMOTE and under-sampling methods, the model balanced its predictions, ensuring it could handle the imbalanced nature of default data.

With this innovative tool, ProsperFinance transformed its loan evaluation process, enabling efficient, accurate, and equitable lending decisions, securing the company’s future while empowering countless borrowers.
    """)

# ========== DATA PROCESSING MODULE ==========
def load_and_preprocess_data():
    """Load, clean, and preprocess the dataset."""
    dfs = [pd.read_csv(f'chunk_{i}.csv') for i in range(10)]
    data = pd.concat(dfs, ignore_index=True)

    # Drop unnecessary columns
    columns_to_remove = [
        "hardship_type", "hardship_reason", "hardship_status", "hardship_start_date",
        "hardship_end_date", "payment_plan_start_date", "hardship_loan_status",
        "verification_status_joint", "sec_app_earliest_cr_line", "next_pymnt_d",
        "earliest_cr_line", "last_credit_pull_d", "revol_util"
    ]
    data.drop(columns=columns_to_remove, inplace=True)

    # Handle missing values
    numerical_cols = data.select_dtypes(include=['number']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    numerical_imputer = SimpleImputer(strategy='mean')
    data[numerical_cols] = numerical_imputer.fit_transform(data[numerical_cols])

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

    # Encode categorical features
    for col in categorical_cols:
        data[col] = pd.factorize(data[col])[0]

    return data
column_description_file = "LCDataDictionary.xlsx" 

# ========== DATA DESCRIPTION MODULE ==========

def show_data_description(data, column_description_file):
    """Displays the dataset overview and column descriptions in Streamlit."""
    
    # Load column descriptions
    column_descriptions = pd.read_excel(column_description_file)

    # Display dataset overview
    st.header("Data Description")
    
    # Dataset Overview
    st.write("### Dataset Overview")
    st.dataframe(data.head())

    # Column Descriptions
    st.write("### Column Descriptions")
    st.dataframe(column_descriptions)

# ========== INITIAL DATA ANALYSIS MODULE ==========


# ========== DATA VISUALIZATION MODULE ==========
def plot_eda(data):
    """Create exploratory data analysis visualizations."""
    st.header("Exploratory Data Analysis")
    st.write("### Distribution of Variables") 
    selected_column = st.selectbox("Select a column to view its distribution", data.select_dtypes(include=['float', 'int']).columns)
    if selected_column:
        st.write(f"Distribution of {selected_column}")
        fig, ax = plt.subplots()
        sns.histplot(data[selected_column], kde=True, ax=ax)
        st.pyplot(fig)
    # for col in data.select_dtypes(include=['float', 'int']).columns: 
    #     st.write(f"Distribution of {col}") 
    #     fig, ax = plt.subplots() 
    #     sns.histplot(data[col], kde=True, ax=ax) 
    #     st.pyplot(fig) 
    # st.subheader("Distribution of Loan Amount")
    # fig, ax = plt.subplots()
    # sns.histplot(data['loan_amnt'], kde=True, ax=ax)
    # ax.set_title("Loan Amount Distribution")
    # st.pyplot(fig)

    # Additional EDA visualizations
    df = data.rename(columns={"loan_amnt": "loan_amount", "funded_amnt": "funded_amount", 
                              "funded_amnt_inv": "investor_funds", "int_rate": "interest_rate", 
                              "annual_inc": "annual_income"}) 
    df.drop(['id', 'emp_title', 'url', 'zip_code', 'title'], axis=1, inplace=True) 

    fig, ax = plt.subplots(1, 3, figsize=(16, 5)) 
    loan_amount = df["loan_amount"].values 
    funded_amount = df["funded_amount"].values 
    investor_funds = df["investor_funds"].values 

    sns.histplot(loan_amount, ax=ax[0], color="#F7522F") 
    ax[0].set_title("Loan Applied by the Borrower", fontsize=14) 
    sns.histplot(funded_amount, ax=ax[1], color="#2F8FF7") 
    ax[1].set_title("Amount Funded by the Lender", fontsize=14) 
    sns.histplot(investor_funds, ax=ax[2], color="#2EAD46") 
    ax[2].set_title("Total committed by Investors", fontsize=14) 
    st.pyplot(fig)

    st.title("Issuance of Loans Over Time")
    st.write("This app visualizes the issuance of loans across different years.")
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%Y-%m-%d', errors='coerce')
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='mixed', errors='coerce')
    # df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df['year'] = df['issue_d'].dt.year 
    fig, ax = plt.subplots(figsize=(12, 8)) 
    sns.barplot(x='year', y='loan_amount', data=df, hue='year', palette='tab10', estimator=sum, legend=True) 
    ax.set_title('Issuance of Loans', fontsize=16) 
    ax.set_xlabel('Year', fontsize=14) 
    ax.set_ylabel('Average loan amount issued', fontsize=14) 
    st.pyplot(fig) 


    st.title("Issuance of Loans and Loan Conditions")
    st.write("This app visualizes the issuance of loans over time and the distribution of loan conditions.")
    bad_loan = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", 
                "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"] 
    df['loan_condition'] = df['loan_status'].apply(lambda status: 'Bad Loan' if status in bad_loan else 'Good Loan') 

    fig, ax = plt.subplots(1, 2, figsize=(16, 8)) 
    colors = ["#3791D7", "#D72626"] 
    labels = ["Good Loans", "Bad Loans"] 
    explode = [0.1] * len(df["loan_condition"].value_counts())
    df["loan_condition"].value_counts().plot.pie(explode=explode, autopct='%1.2f%%', ax=ax[0],
                                                 shadow=True, colors=colors, labels=labels, fontsize=12, startangle=70) 
    ax[0].set_ylabel('% of Condition of Loans', fontsize=14) 
    sns.barplot(x="year", y="loan_amount", hue="loan_condition", data=df, 
                palette=["#3791D7", "#E01E1B"], estimator=lambda x: len(x) / len(df) * 100, ax=ax[1]) 
    ax[1].set(ylabel="(%)") 
    st.pyplot(fig) 

    st.subheader("Summary for types of loans")
    st.write("The NorthEast region seems to be the most attractive in terms of funding loans to borrowers.")
    st.write("The SouthWest and West regions have experienced a slight increase in the 'median income' in the past years.")
    st.write("Average interest rates have declined since 2012, but this might explain the increase in the volume of loans.")
    st.write("Employment Length tends to be greater in the regions of the SouthWest and West.")
    st.write("Clients located in the regions of NorthEast and MidWest have not experienced a drastic increase in debt-to-income (dti) as compared to the other regions.")


    west = ['CA', 'OR', 'UT', 'WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID'] 
    south_west = ['AZ', 'TX', 'NM', 'OK'] 
    south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN'] 
    mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND'] 
    north_east = ['CT', 'NY', 'PA', 'NJ', 'RI', 'MA', 'MD', 'VT', 'NH', 'ME'] 
    df['region'] = df['addr_state'].apply(lambda state: 'West' if state in west else 'SouthWest' if state in south_west 
                                          else 'SouthEast' if state in south_east else 'MidWest' if state in mid_west else 'NorthEast') 

    group_dates = df.groupby(['issue_d', 'region'], as_index=False).sum(numeric_only=True) 
    group_dates['loan_amount'] = group_dates['loan_amount'] / 1000 
    df_dates = pd.DataFrame(data=group_dates[['issue_d', 'region', 'loan_amount']]) 

    # fig, ax = plt.subplots(figsize=(15, 6)) 
    # by_issued_amount = df_dates.groupby(['issue_d', 'region']).loan_amount.sum() 
    # by_issued_amount.unstack().plot(stacked=False, colormap=plt.cm.Set3, grid=False, legend=True, ax=ax) 
    # ax.set_title('Loans issued by Region', fontsize=16) 
    # st.pyplot(fig)
    # Summary section
    st.subheader("Summary of loans by region")
    st.write("SouthEast, West, and NorthEast regions had the highest amount of loans issued.")
    st.write("West and SouthWest had a rapid increase in debt-to-income starting in 2012.")
    st.write("West and SouthWest had a rapid decrease in interest rates. This might explain the increase in debt-to-income.")


    st.subheader("Loan Conditions by Region from the actual data")
    loan_conditions_table = {
        "region": ["MidWest", "NorthEast", "SouthEast", "SouthWest", "West"],
        "Charged Off": [7361, 10671, 11094, 4774, 11348],
        "Default": [175, 263, 297, 166, 318],
        "Does not meet the credit policy. Status: Charged Off": [142, 190, 184, 79, 166],
        "In Grace Period": [926, 1625, 1579, 708, 1415],
        "Late (16-30 days)": [354, 585, 600, 273, 545],
        "Late (31-120 days)": [1820, 2799, 2925, 1407, 2640],
        "Total": [10778, 16133, 16679, 7407, 16432]
    }
    df_table = pd.DataFrame(loan_conditions_table)
    st.dataframe(df_table)


    df['emp_length_int'] = 0
    df['interest_rate'] = pd.to_numeric(df['interest_rate'], errors='coerce')
    df['interest_rate'] = df['interest_rate'].astype(int)
    df['emp_length_int'] = pd.to_numeric(df['emp_length_int'], errors='coerce')
    df['emp_length_int'] = df['emp_length_int'].astype(int)
    df['dti'] = pd.to_numeric(df['dti'], errors='coerce')
    df['dti'] = df['dti'].astype(int)
    df['annual_income'] = pd.to_numeric(df['annual_income'], errors='coerce')
    df['annual_income'] = df['annual_income'].astype(int)
    for col in [df]: 
        col.loc[col['emp_length'] == '10+ years', "emp_length_int"] = 10 
        col.loc[col['emp_length'] == '9 years', "emp_length_int"] = 9 
        col.loc[col['emp_length'] == '8 years', "emp_length_int"] = 8 
        col.loc[col['emp_length'] == '7 years', "emp_length_int"] = 7 
        col.loc[col['emp_length'] == '6 years', "emp_length_int"] = 6 
        col.loc[col['emp_length'] == '5 years', "emp_length_int"] = 5 
        col.loc[col['emp_length'] == '4 years', "emp_length_int"] = 4 
        col.loc[col['emp_length'] == '3 years', "emp_length_int"] = 3 
        col.loc[col['emp_length'] == '2 years', "emp_length_int"] = 2 
        col.loc[col['emp_length'] == '1 year', "emp_length_int"] = 1 
        col.loc[col['emp_length'] == '< 1 year', "emp_length_int"] = 0.5 
        col.loc[col['emp_length'] == 'n/a', "emp_length_int"] = 0 

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # by_interest_rate = df.groupby(['year', 'region']).interest_rate.mean() 
    # by_interest_rate.unstack().plot(kind='area', stacked=True, colormap=plt.cm.inferno, grid=False, legend=False, ax=ax1) 
    # ax1.set_title('Average Interest Rate by Region', fontsize=14)

    # by_employment_length = df.groupby(['year', 'region']).emp_length_int.mean() 
    # by_employment_length.unstack().plot(kind='area', stacked=True, colormap=plt.cm.inferno, grid=False, legend=False, ax=ax2)

    # by_dti = df.groupby(['year', 'region']).dti.mean() 
    # by_dti.unstack().plot(kind='area', stacked=True, colormap=plt.cm.cool, grid=False, legend=False, ax=ax3) 
    # ax3.set_title('Average Debt-to-Income by Region', fontsize=14)

    # by_income = df.groupby(['year', 'region']).annual_income.mean() 
    # by_income.unstack().plot(kind='area', stacked=True, colormap=plt.cm.cool, grid=False, ax=ax4) 
    # ax4.set_title('Average Annual Income by Region', fontsize=14)
    # ax4.legend(bbox_to_anchor=(-1.0, -0.5, 1.8, 0.1), loc=10, prop={'size': 12}, ncol=5, mode="expand", borderaxespad=0.)

    # st.pyplot(fig)

def show_correlation_heatmap(data):
    st.header("Correlation Heatmap")
    st.write("### Correlation Matrix")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def show_top_correlation_heatmap(data):
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    st.header("Top 10 Highly Correlated Features Heatmap")
    st.write("### Correlation Matrix")

    # Compute the correlation matrix
    corr = data.corr()

    # Flatten the correlation matrix and filter for the top 10 correlations (excluding self-correlations)
    corr_unstacked = corr.abs().unstack()
    corr_unstacked = corr_unstacked[corr_unstacked < 1.0]  # Exclude self-correlations
    top_corrs = corr_unstacked.nlargest(10)

    # Extract the indices of the top 10 correlations
    top_corr_features = pd.Index(top_corrs.index.get_level_values(0))
    top_corr_features = top_corr_features.union(pd.Index(top_corrs.index.get_level_values(1)))

    # Subset the correlation matrix to include only top features
    top_corr_matrix = corr.loc[top_corr_features, top_corr_features]

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(top_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

    # Display the heatmap in Streamlit
    st.pyplot(fig)

# def show_model_parameters(models):
#     st.header("Model Parameters")
#     for model_name, model in models.items():
#         st.subheader(model_name)
#         st.json(model.get_params())

def show_model_parameters(models):
    st.header("Model Parameters")
    selected_model = st.selectbox("Select a model", list(models.keys()))
    model = models[selected_model]
    st.subheader(f"{selected_model} Parameters")
    st.json(model.get_params())

# def evaluate_model(model, X_train, y_train, X_test, y_test):
#     st.subheader(f"Evaluation: {model.__class__.__name__}")
#     try:
#         # Train the model
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         st.text(classification_report(y_test, y_pred))

#         if hasattr(model, "predict_proba"):
#             y_proba = model.predict_proba(X_test)
#             # Check for binary or multiclass
#             if len(np.unique(y_test)) == 2:
#                 fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
#             else:
#                 # Handle multiclass by selecting one class
#                 class_index = 1  # Change this to the index of the class to evaluate
#                 y_test_binary = label_binarize(y_test, classes=np.unique(y_test))[:, class_index]
#                 fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, class_index])

#             roc_auc = auc(fpr, tpr)
#             fig, ax = plt.subplots()
#             ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
#             ax.plot([0, 1], [0, 1], 'k--')
#             ax.set_title("ROC Curve")
#             ax.set_xlabel("False Positive Rate")
#             ax.set_ylabel("True Positive Rate")
#             ax.legend()
#             st.pyplot(fig)
#         else:
#             st.warning("Model does not support probability predictions.")

#     except NotFittedError:
#         st.error("The model is not fitted. Check the training process.")
#     except ValueError as e:
#         st.error(f"ValueError: {e}")


def evaluate_models(models, X_train, y_train, X_test, y_test):
    # Debugging: Check if 'models' is a dictionary
    if not isinstance(models, dict):
        st.error("Expected 'models' to be a dictionary.")
        print("models is of type:", type(models))  # Print type of models to the console
        return

    selected_models = st.multiselect("Select models to evaluate", list(models.keys()))
    
    if selected_models:
        for model_name in selected_models:
            model = models[model_name]
            st.subheader(f"Evaluation: {model_name}")
            try:
                # Train the model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.text(classification_report(y_test, y_pred))

                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                    # Check for binary or multiclass
                    if len(np.unique(y_test)) == 2:
                        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    else:
                        # Handle multiclass by selecting one class
                        class_index = 1  # Change this to the index of the class to evaluate
                        y_test_binary = label_binarize(y_test, classes=np.unique(y_test))[:, class_index]
                        fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, class_index])

                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_title(f"{model_name} ROC Curve")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning(f"{model_name} does not support probability predictions.")
            
            except Exception as e:
                st.error(f"Error evaluating {model_name}: {e}")
    else:
        st.warning("Please select at least one model to evaluate.")

# ----------- Main App Logic ----------- #
# data = load_data()
# data = preprocess_data(data)
data = load_and_preprocess_data()

X = data.drop("loan_status", axis=1)
y = pd.factorize(data["loan_status"])[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "Gradient Boosting": GradientBoostingClassifier(random_state=42)
# }

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

tabs = ["Problem Statement", "Data Description","Initial Data Analysis", "EDA", "Correlation Heatmap", "Model Parameters", "Model Evaluation"]
selected_tab = st.sidebar.radio("Navigation", tabs)

if selected_tab == "Problem Statement":
    show_problem_statement()
elif selected_tab == "Data Description":
    show_data_description(data,column_description_file)
elif selected_tab == "Initial Data Analysis":
    st.header("Initial Data Analysis")
    st.write("### Summary Statistics")
    st.dataframe(data.describe())


    st.subheader("Columns with Missing Values")
    missing_data = {
        "Column": ["emp_title", "emp_length", "title", "zip_code", "last_pymnt_d", "hardship_flag"],
        "Missing Count": [197785, 153833, 17500, 1, 3666, 28873]
    }
    df_missing = pd.DataFrame(missing_data)
    st.dataframe(df_missing)
    st.subheader("Handling Missing Data")
    st.write("""
    - **Numerical Features**: Missing values in numerical columns were imputed using the mean of the respective column.
    - **Categorical Features**: Missing values in categorical columns were imputed using the most frequent value in the respective column.
    This ensures that the dataset remains consistent and minimizes the loss of information due to missing values.
    """)
elif selected_tab == "EDA":
    plot_eda(data)
elif selected_tab == "Correlation Heatmap":
    show_top_correlation_heatmap(data)
elif selected_tab == "Model Parameters":
    show_model_parameters(models)
elif selected_tab == "Model Evaluation":
    for model_name, model in models.items():
        evaluate_models(models, X_train, y_train, X_test, y_test)

