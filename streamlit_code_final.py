import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
from collections import OrderedDict
import warnings

# Load data and column descriptions.
warnings.filterwarnings("ignore")
# Load data and column descriptions.
# data_file = """D:\Foundations_of_DataScience\Projects\Final_projects\lending_club_dataset_2007_2020\lending_club_clean.feather"""  # Replace with your CSV file path
column_description_file = "LCDataDictionary.xlsx" # Replace with your column description CSV file path
dfs = []
for i in range(9):
    file_path = os.path.join(directory, f'chunk_{i}.csv')
    df = pd.read_csv(file_path)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
df=data.copy()

data = pd.read_feather(data_file)

columns_to_remove = [
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "hardship_loan_status",
    "verification_status_joint",
    "sec_app_earliest_cr_line",
    "next_pymnt_d",
    "earliest_cr_line",
    "last_credit_pull_d",
    "revol_util"
]
data = data.drop(columns=columns_to_remove)

data = data[(~data['emp_length'].isna()) & (~data['emp_title'].isna())].copy()
data = data.dropna(subset=['zip_code'])
numerical_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object', 'category']).columns
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = numerical_imputer.fit_transform(data[numerical_cols])
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = pd.factorize(data[col])[0]
column_descriptions = pd.read_csv(column_description_file)

# Splitting the dataset into X and y
X = data.drop("loan_status", axis=1)
y = data["loan_status"]
y = pd.factorize(y)[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scaling the features
scaler = StandardScaler()
X_train = X_train.select_dtypes(include=['number'])
X_test = X_test.select_dtypes(include=['number'])
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Under-sampling the data to address class imbalance
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Define model evaluation function
def evaluate_model_multiclass(model, X_train, y_train, X_test, y_test):
    y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_train_bin.shape[1]
    
    classifier = OneVsRestClassifier(model)
    classifier.fit(X_train, y_train_bin)
    
    y_proba = classifier.predict_proba(X_test)
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (area = {roc_auc[i]:0.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Multiclass")
    st.pyplot(plt)
    
    y_pred = classifier.predict(X_test)
    st.write("Classification Report:")
    st.text(classification_report(y_test_bin, y_pred))

# Define models
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

# Streamlit App
st.title("Project Dashboard")

# Tabs
tabs = ["Problem Statement", "Data Description", "Initial Data Analysis", 
        "Exploratory Data Analysis","Correlation Heatmaps", "Model Parameters", "Model Evaluation", 
        "Conclusion", "Future Scope"]

selected_tab = st.sidebar.radio("Navigate", tabs)

# Tab: Problem Statement
if selected_tab == "Problem Statement":
    st.header("Problem Statement")
    st.write("""Problem Statement Story: Lending Risk Analysis
In a bustling city, a lending institution named ProsperFinance was thriving, helping individuals achieve their dreams by providing loans for homes, education, and businesses. However, as the business grew, ProsperFinance faced a daunting challenge: how to effectively identify borrowers who might pose a higher risk of defaulting on their loans. Each default not only impacted the company financially but also its ability to extend opportunities to other borrowers.

The company’s decision-makers realized they were sitting on a treasure trove of data. They had records of previous applicants, including their age, income, homeownership status, loan amount requested, credit history, payment behavior, and past loan performance. However, manually sifting through this data was inefficient, subjective, and prone to error.

Recognizing the need for a robust solution, ProsperFinance sought to develop a cutting-edge web application to automate this process. The app needed to analyze the data, identify patterns, and predict the risk associated with new loan applicants. The ultimate goal was twofold: to safeguard the company’s resources and maintain fairness in lending by basing decisions on clear, data-driven insights.

To achieve this, a team of data scientists began by conducting thorough exploratory data analysis (EDA) to uncover key factors influencing loan defaults. They then engineered features like debt-to-income ratios and payment consistency to enhance predictive accuracy. Leveraging advanced algorithms such as Gradient Boosting, Support Vector Machines (SVM), Artificial Neural Networks (ANN), and Optimally Weighted Fuzzy K-Nearest Neighbors (OWFKNN), they designed a comprehensive risk evaluation system.

The project culminated in a user-friendly Streamlit dashboard. This app not only provided intuitive risk predictions but also offered a behind-the-scenes view of the data analysis and modeling process for stakeholders, ensuring trust and transparency. By integrating oversampling techniques like SMOTE and under-sampling methods, the model balanced its predictions, ensuring it could handle the imbalanced nature of default data.

With this innovative tool, ProsperFinance transformed its loan evaluation process, enabling efficient, accurate, and equitable lending decisions, securing the company’s future while empowering countless borrowers.
    """)

# Tab: Data Description
elif selected_tab == "Data Description":
    st.header("Data Description")
    st.write("### Dataset Overview")
    st.dataframe(data.head())

    st.write("### Column Descriptions")
    st.dataframe(column_descriptions)

# Tab: Initial Data Analysis
elif selected_tab == "Initial Data Analysis":
    st.header("Initial Data Analysis")
    st.write("### Summary Statistics")
    st.dataframe(data.describe())

    st.write("### Missing Values")
    st.write(data.isnull().sum())

# Tab: Exploratory Data Analysis
elif selected_tab == "Exploratory Data Analysis": 
    st.header("Exploratory Data Analysis") 
    st.write("### Distribution of Variables") 
    for col in data.select_dtypes(include=['float', 'int']).columns: 
        st.write(f"Distribution of {col}") 
        fig, ax = plt.subplots() 
        sns.histplot(data[col], kde=True, ax=ax) 
        st.pyplot(fig) 

    # Additional EDA Code 
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

    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y') 
    df['year'] = df['issue_d'].dt.year 
    fig, ax = plt.subplots(figsize=(12, 8)) 
    sns.barplot(x='year', y='loan_amount', data=df, hue='year', palette='tab10', estimator=sum, legend=True) 
    ax.set_title('Issuance of Loans', fontsize=16) 
    ax.set_xlabel('Year', fontsize=14) 
    ax.set_ylabel('Average loan amount issued', fontsize=14) 
    st.pyplot(fig) 

    bad_loan = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", 
                "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"] 
    df['loan_condition'] = df['loan_status'].apply(lambda status: 'Bad Loan' if status in bad_loan else 'Good Loan') 

    fig, ax = plt.subplots(1, 2, figsize=(16, 8)) 
    colors = ["#3791D7", "#D72626"] 
    labels = ["Good Loans", "Bad Loans"] 
    df["loan_condition"].value_counts().plot.pie(explode=[0, 0.25], autopct='%1.2f%%', ax=ax[0], 
                                                 shadow=True, colors=colors, labels=labels, fontsize=12, startangle=70) 
    ax[0].set_ylabel('% of Condition of Loans', fontsize=14) 
    sns.barplot(x="year", y="loan_amount", hue="loan_condition", data=df, 
                palette=["#3791D7", "#E01E1B"], estimator=lambda x: len(x) / len(df) * 100, ax=ax[1]) 
    ax[1].set(ylabel="(%)") 
    st.pyplot(fig) 

    # Define regions based on states 
    west = ['CA', 'OR', 'UT', 'WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID'] 
    south_west = ['AZ', 'TX', 'NM', 'OK'] 
    south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN'] 
    mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND'] 
    north_east = ['CT', 'NY', 'PA', 'NJ', 'RI', 'MA', 'MD', 'VT', 'NH', 'ME'] 
    df['region'] = df['addr_state'].apply(lambda state: 'West' if state in west else 'SouthWest' if state in south_west 
                                          else 'SouthEast' if state in south_east else 'MidWest' if state in mid_west else 'NorthEast') 

    df['complete_date'] = pd.to_datetime(df['issue_d'], format='%b-%Y') 
    group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum(numeric_only=True) 
    group_dates['issue_d'] = group_dates['complete_date'].dt.to_period('M') 
    group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum(numeric_only=True) 
    group_dates['loan_amount'] = group_dates['loan_amount'] / 1000 
    df_dates = pd.DataFrame(data=group_dates[['issue_d', 'region', 'loan_amount']]) 

    fig, ax = plt.subplots(figsize=(15, 6)) 
    by_issued_amount = df_dates.groupby(['issue_d', 'region']).loan_amount.sum() 
    by_issued_amount.unstack().plot(stacked=False, colormap=plt.cm.Set3, grid=False, legend=True, ax=ax) 
    ax.set_title('Loans issued by Region', fontsize=16) 
    st.pyplot(fig)

        # Employment length processing
    df['emp_length_int'] = np.nan 
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

    # Visualize additional insights
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    by_interest_rate = df.groupby(['year', 'region']).interest_rate.mean() 
    by_interest_rate.unstack().plot(kind='area', stacked=True, colormap=plt.cm.inferno, grid=False, legend=False, ax=ax1) 
    ax1.set_title('Average Interest Rate by Region', fontsize=14)

    by_employment_length = df.groupby(['year', 'region']).emp_length_int.mean() 
    by_employment_length.unstack().plot(kind='area', stacked=True, colormap=plt.cm.inferno, grid=False, legend=False, ax=ax2)

    by_dti = df.groupby(['year', 'region']).dti.mean() 
    by_dti.unstack().plot(kind='area', stacked=True, colormap=plt.cm.cool, grid=False, legend=False, ax=ax3) 
    ax3.set_title('Average Debt-to-Income by Region', fontsize=14)

    by_income = df.groupby(['year', 'region']).annual_income.mean() 
    by_income.unstack().plot(kind='area', stacked=True, colormap=plt.cm.cool, grid=False, ax=ax4) 
    ax4.set_title('Average Annual Income by Region', fontsize=14)
    ax4.legend(bbox_to_anchor=(-1.0, -0.5, 1.8, 0.1), loc=10, prop={'size': 12}, ncol=5, mode="expand", borderaxespad=0.)

    st.pyplot(fig)

    # Assuming the previous data processing code and library imports are already included

# Filtering the DataFrame to include only bad loans
badloans_df = df.loc[df["loan_condition"] == "Bad Loan"]

# Creating a crosstab of loan status by region and calculating percentages
loan_status_cross = pd.crosstab(badloans_df['region'], badloans_df['loan_status']).apply(lambda x: x/x.sum() * 100)

# Creating a crosstab of the number of loan statuses by region
number_of_loanstatus = pd.crosstab(badloans_df['region'], badloans_df['loan_status'])

# Rounding the percentage values to two decimal places
loan_status_cross['Charged Off'] = loan_status_cross['Charged Off'].apply(lambda x: round(x, 2))
loan_status_cross['Default'] = loan_status_cross['Default'].apply(lambda x: round(x, 2))
loan_status_cross['Does not meet the credit policy. Status:Charged Off'] = loan_status_cross['Does not meet the credit policy. Status:Charged Off'].apply(lambda x: round(x, 2))
loan_status_cross['In Grace Period'] = loan_status_cross['In Grace Period'].apply(lambda x: round(x, 2))
loan_status_cross['Late (16-30 days)'] = loan_status_cross['Late (16-30 days)'].apply(lambda x: round(x, 2))
loan_status_cross['Late (31-120 days)'] = loan_status_cross['Late (31-120 days)'].apply(lambda x: round(x, 2))

# Calculating the total number of loan statuses by region
number_of_loanstatus['Total'] = number_of_loanstatus.sum(axis=1)

# Extracting data for each loan status category
charged_off = loan_status_cross['Charged Off'].values.tolist()
default = loan_status_cross['Default'].values.tolist()
not_meet_credit = loan_status_cross['Does not meet the credit policy. Status:Charged Off'].values.tolist()
grace_period = loan_status_cross['In Grace Period'].values.tolist()
short_pay = loan_status_cross['Late (16-30 days)'].values.tolist()
long_pay = loan_status_cross['Late (31-120 days)'].values.tolist()

# Creating bar plots for each loan status
charged = go.Bar(
    x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y=charged_off,
    name='Charged Off',
    marker=dict(
        color='rgb(192, 148, 246)'
    ),
    text='%'
)

defaults = go.Bar(
    x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y=default,
    name='Defaults',
    marker=dict(
        color='rgb(176, 26, 26)'
    ),
    text='%'
)

credit_policy = go.Bar(
    x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y=not_meet_credit,
    name='Does not meet Credit Policy',
    marker=dict(
        color='rgb(229, 121, 36)'
    ),
    text='%'
)

grace = go.Bar(
    x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y=grace_period,
    name='Grace Period',
    marker=dict(
        color='rgb(147, 147, 147)'
    ),
    text='%'
)

short_pays = go.Bar(
    x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y=short_pay,
    name='Late Payment (16-30 days)',
    marker=dict(
        color='rgb(246, 157, 135)'
    ),
    text='%'
)

long_pays = go.Bar(
    x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y=long_pay,
    name='Late Payment (31-120 days)',
    marker=dict(
        color='rgb(238, 76, 73)'
    ),
    text='%'
)

# Assembling data for the plot
data = [charged, defaults, credit_policy, grace, short_pays, long_pays]

# Defining the layout of the plot
layout = go.Layout(
    barmode='stack',
    title='% of Bad Loan Status by Region',
    xaxis=dict(title='US Regions')
)

# Creating the figure and plotting
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stacked-bar')


# Assuming the previous code is defined above this point, add the following visualization under the EDA tab:

# Import necessary libraries for plotting
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import OrderedDict

# Group data for visualization
by_loan_amount = df.groupby(['region','addr_state'], as_index=False).loan_amount.sum()
by_interest_rate = df.groupby(['region', 'addr_state'], as_index=False).interest_rate.mean()
by_income = df.groupby(['region', 'addr_state'], as_index=False).annual_income.mean()
states = by_loan_amount['addr_state'].values.tolist()
average_loan_amounts = by_loan_amount['loan_amount'].values.tolist()
average_interest_rates = by_interest_rate['interest_rate'].values.tolist()
average_annual_income = by_income['annual_income'].values.tolist()

# Create a DataFrame for metrics
metrics_data = OrderedDict([('state_codes', states),
                            ('issued_loans', average_loan_amounts),
                            ('interest_rate', average_interest_rates),
                            ('annual_income', average_annual_income)])
metrics_df = pd.DataFrame.from_dict(metrics_data)
metrics_df = metrics_df.round(decimals=2)

# Ensure all columns are strings for Plotly plotting
for col in metrics_df.columns:
    metrics_df[col] = metrics_df[col].astype(str)

# Define color scale for the choropleth map
scl = [[0.0, 'rgb(210, 241, 198)'],[0.2, 'rgb(188, 236, 169)'],[0.4, 'rgb(171, 235, 145)'],
       [0.6, 'rgb(140, 227, 105)'],[0.8, 'rgb(105, 201, 67)'],[1.0, 'rgb(59, 159, 19)']]

# Add text to display in the map
metrics_df['text'] = metrics_df['state_codes'] + '<br>' + \
                     metrics_df['default_ratio'].astype(str) + '<br>' + \
                     metrics_df['badloans_amount'].astype(str) + '<br>' + \
                     metrics_df['percentage_of_badloans'].astype(str) + '<br>' + \
                     metrics_df['average_dti'].astype(str) + '<br>' + \
                     metrics_df['average_emp_length'].astype(str)

# Create choropleth map for issued loans
data = [dict(
    type='choropleth',
    colorscale=scl,
    autocolorscale=False,
    locations=metrics_df['state_codes'],
    z=metrics_df['issued_loans'], 
    locationmode='USA-states',
    text=metrics_df['text'],
    marker=dict(
        line=dict(
            color='rgb(255,255,255)',
            width=2
        )
    ),
    colorbar=dict(
        title="$s USD"
    )
)]

layout = dict(
    title='Issued Loans',
    geo=dict(
        scope='usa',
        projection=dict(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'
    )
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='d3-cloropleth-map')

# Categorize income into low, medium, and high
for col in lst:
    col.loc[col['annual_income'] <= 100000, 'income_category'] = 'Low'
    col.loc[(col['annual_income'] > 100000) & (col['annual_income'] <= 200000), 'income_category'] = 'Medium'
    col.loc[col['annual_income'] > 200000, 'income_category'] = 'High'

# Assign loan condition integer values
for col in lst:
    col.loc[df['loan_condition'] == 'Good Loan', 'loan_condition_int'] = 0  # Negative (Bad Loan)
    col.loc[df['loan_condition'] == 'Bad Loan', 'loan_condition_int'] = 1  # Positive (Good Loan)

df['loan_condition_int'] = df['loan_condition_int'].astype(int)

# Create subplots to show visualizations for loans by credit score and interest rate
f, ((ax1, ax2)) = plt.subplots(1, 2)
cmap = plt.cm.coolwarm

# Plot loans issued by credit score
by_credit_score = df.groupby(['year', 'grade']).loan_amount.mean()
by_credit_score.unstack().plot(legend=False, ax=ax1, figsize=(14, 4), colormap=cmap)
ax1.set_title('Loans issued by Credit Score', fontsize=14)

# Plot interest rates by credit score
by_inc = df.groupby(['year', 'grade']).interest_rate.mean()
by_inc.unstack().plot(ax=ax2, figsize=(14, 4), colormap=cmap)
ax2.set_title('Interest Rates by Credit Score', fontsize=14)

ax2.legend(bbox_to_anchor=(-1.0, -0.3, 1.7, 0.1), loc=5, prop={'size': 12},
           ncol=7, mode="expand", borderaxespad=0.)

# Create a figure for the next set of subplots
fig = plt.figure(figsize=(16, 12))

# Add subplots for type of loans by grade, sub-grade, and average interest rate by loan condition
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(212)

cmap = plt.cm.coolwarm_r

# Plot type of loans by grade
loans_by_region = df.groupby(['grade', 'loan_condition']).size()
loans_by_region.unstack().plot(kind='bar', stacked=True, colormap=cmap, ax=ax1, grid=False)
ax1.set_title('Type of Loans by Grade', fontsize=14)

# Plot type of loans by sub-grade
loans_by_grade = df.groupby(['sub_grade', 'loan_condition']).size()
loans_by_grade.unstack().plot(kind='bar', stacked=True, colormap=cmap, ax=ax2, grid=False)
ax2.set_title('Type of Loans by Sub-Grade', fontsize=14)

# Plot average interest rate by loan condition
by_interest = df.groupby(['year', 'loan_condition']).interest_rate.mean()
by_interest.unstack().plot(ax=ax3, colormap=cmap)
ax3.set_title('Average Interest rate by Loan Condition', fontsize=14)
ax3.set_ylabel('Interest Rate (%)', fontsize=12)

# Plot time series data for different loan statuses (bad loans)
numeric_variables = df.select_dtypes(exclude=["object"])

# Title and labels for bad loan plots
title = 'Bad Loans: Loan Statuses'

labels = bad_loan  # All the elements that comprise a bad loan.

colors = ['rgba(236, 112, 99, 1)', 'rgba(235, 152, 78, 1)', 'rgba(52, 73, 94, 1)', 'rgba(128, 139, 150, 1)',
          'rgba(255, 87, 51, 1)', 'rgba(255, 195, 0, 1)']

mode_size = [8, 8, 8, 8, 8, 8]
line_size = [2, 2, 2, 2, 2, 2]

x_data = [
    sorted(df['year'].unique().tolist()),
    sorted(df['year'].unique().tolist()),
    sorted(df['year'].unique().tolist()),
    sorted(df['year'].unique().tolist()),
    sorted(df['year'].unique().tolist()),
    sorted(df['year'].unique().tolist()),
]

# Extract data for plotting
charged_off = df['loan_amount'].loc[df['loan_status'] == 'Charged Off'].values.tolist()
defaults = df['loan_amount'].loc[df['loan_status'] == 'Default'].values.tolist()
not_credit_policy = df['loan_amount'].loc[df['loan_status'] == 'Does not meet the credit policy. Status:Charged Off'].values.tolist()
grace_period = df['loan_amount'].loc[df['loan_status'] == 'In Grace Period'].values.tolist()
short_late = df['loan_amount'].loc[df['loan_status'] == 'Late (16-30 days)'].values.tolist()
long_late = df['loan_amount'].loc[df['loan_status'] == 'Late (31-120 days)'].values.tolist()

# Create traces for each loan status
p_charged_off = go.Scatter(
    x=x_data[0],
    y=charged_off,
    name='A. Charged Off',
    line=dict(
        color=colors[0],
        width=3,
        dash='dash'
    )
)


# Function to prepare data for metrics DataFrame
def prepare_metrics_df(df):
    """Prepare a DataFrame containing average loan amount, interest rate, and annual income grouped by region and state."""
    by_loan_amount = df.groupby(['region', 'addr_state'], as_index=False).loan_amount.sum()
    by_interest_rate = df.groupby(['region', 'addr_state'], as_index=False).interest_rate.mean()
    by_income = df.groupby(['region', 'addr_state'], as_index=False).annual_income.mean()

    states = by_loan_amount['addr_state'].values.tolist()
    average_loan_amounts = by_loan_amount['loan_amount'].values.tolist()
    average_interest_rates = by_interest_rate['interest_rate'].values.tolist()
    average_annual_income = by_income['annual_income'].values.tolist()

    metrics_data = OrderedDict([
        ('state_codes', states),
        ('issued_loans', average_loan_amounts),
        ('interest_rate', average_interest_rates),
        ('annual_income', average_annual_income)
    ])
    metrics_df = pd.DataFrame.from_dict(metrics_data)
    metrics_df = metrics_df.round(decimals=2)

    # Ensure all columns are strings for Plotly plotting
    for col in metrics_df.columns:
        metrics_df[col] = metrics_df[col].astype(str)

    return metrics_df

# Function to create a choropleth map
def create_choropleth_map(metrics_df):
    """Create a choropleth map to visualize issued loans by state."""
    # Define color scale for the choropleth map
    scl = [[0.0, 'rgb(210, 241, 198)'], [0.2, 'rgb(188, 236, 169)'], [0.4, 'rgb(171, 235, 145)'],
           [0.6, 'rgb(140, 227, 105)'], [0.8, 'rgb(105, 201, 67)'], [1.0, 'rgb(59, 159, 19)']]

    # Add text to display in the map
    metrics_df['text'] = (metrics_df['state_codes'] + '<br>' +
                        'Average loan interest rate: ' + metrics_df['interest_rate'] + '<br>' +
                        'Average annual income: ' + metrics_df['annual_income'])

    data = [dict(
        type='choropleth',
        colorscale=scl,
        autocolorscale=False,
        locations=metrics_df['state_codes'],
        z=metrics_df['issued_loans'], 
        locationmode='USA-states',
        text=metrics_df['text'],
        marker=dict(
            line=dict(
                color='rgb(255,255,255)',
                width=2
            )
        ),
        colorbar=dict(
            title="$s USD"
        )
    )]

    layout = dict(
        title='Issued Loans',
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'
        )
    )

    fig = dict(data=data, layout=layout)
    return fig

# Function to categorize income
def categorize_income(df):
    """Categorize income into low, medium, and high."""
    df.loc[df['annual_income'] <= 100000, 'income_category'] = 'Low'
    df.loc[(df['annual_income'] > 100000) & (df['annual_income'] <= 200000), 'income_category'] = 'Medium'
    df.loc[df['annual_income'] > 200000, 'income_category'] = 'High'
    return df

# Function to encode loan condition
def encode_loan_condition(df):
    """Encode the loan condition into integers for analysis."""
    df.loc[df['loan_condition'] == 'Good Loan', 'loan_condition_int'] = 0  # Negative (Bad Loan)
    df.loc[df['loan_condition'] == 'Bad Loan', 'loan_condition_int'] = 1  # Positive (Good Loan)
    df['loan_condition_int'] = df['loan_condition_int'].astype(int)
    return df

# Function to create subplots for loan data visualization
def create_subplots(df):
    """Create subplots to show visualizations for loans by credit score and interest rate."""
    f, ((ax1, ax2)) = plt.subplots(1, 2)
    cmap = plt.cm.coolwarm

    # Plot loans issued by credit score
    by_credit_score = df.groupby(['year', 'grade']).loan_amount.mean()
    by_credit_score.unstack().plot(legend=False, ax=ax1, figsize=(14, 4), colormap=cmap)
    ax1.set_title('Loans issued by Credit Score', fontsize=14)

    # Plot interest rates by credit score
    by_inc = df.groupby(['year', 'grade']).interest_rate.mean()
    by_inc.unstack().plot(ax=ax2, figsize=(14, 4), colormap=cmap)
    ax2.set_title('Interest Rates by Credit Score', fontsize=14)

    ax2.legend(bbox_to_anchor=(-1.0, -0.3, 1.7, 0.1), loc=5, prop={'size': 12},
               ncol=7, mode="expand", borderaxespad=0.)

    return f

# Main code execution
if __name__ == "__main__":
    # Assuming df is already defined and preprocessed
    metrics_df = prepare_metrics_df(df)
    choropleth_fig = create_choropleth_map(metrics_df)
    
    # Display the choropleth map (requires Plotly to run in a Jupyter Notebook or interactive environment)
    from plotly.offline import iplot
    iplot(choropleth_fig, filename='d3-cloropleth-map')

    df = categorize_income(df)
    df = encode_loan_condition(df)

    f = create_subplots(df)
    plt.show()



# Tab Correlation Heatmaps
elif selected_tab == "Correlation Heatmaps":
    st.header("Correlation Heatmaps")

    # Calculate correlations
    correlation = data.corr()

    # Correlation with 'loan_status'
    threshold = 0.5
    loan_status_correlation = correlation["loan_status"].abs()
    highly_correlated = loan_status_correlation[loan_status_correlation > threshold].sort_values(ascending=False)
    st.write("Highly correlated features with 'loan_status':")
    st.write(highly_correlated)

    # Correlation heatmap for features highly correlated with 'loan_status'
    high_corr_features = highly_correlated.index.tolist()
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[high_corr_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap of Highly Correlated Features with 'loan_status'")
    st.pyplot(plt)

    # Apply a higher threshold for filtering correlations
    threshold = 0.9
    np.fill_diagonal(correlation.values, 0)
    filtered_correlation = correlation[correlation.abs() > threshold]
    filtered_correlation = filtered_correlation.dropna(how="all").dropna(axis=1, how="all")

    high_corr_cols = correlation.columns[(correlation.abs() > threshold).any()].tolist()
    if 'loan_status' not in high_corr_cols:
        high_corr_cols.append('loan_status')

    irrelevant_fields = ['id']
    high_corr_cols = [col for col in high_corr_cols if col not in irrelevant_fields]
    filtered_data = data[high_corr_cols]

    st.write(f"Columns kept (highly correlated): {high_corr_cols}")
    filtered_correlation = filtered_data.corr()

    # Correlation heatmap for filtered data with high correlation
    plt.figure(figsize=(14, 10))
    sns.heatmap(filtered_correlation, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Heatmap of Correlations for Filtered Data")
    st.pyplot(plt)

# Tab: Model Parameters
elif selected_tab == "Model Parameters":
    st.title("Model Overview and Parameters")
    for model_name, model in models.items():
        st.subheader(f"{model_name} Parameters")
        st.write(model.get_params())

# Tab: Model Evaluation
elif selected_tab == "Model Evaluation":
    st.title("Model Evaluation")

    # Model Evaluation on original data
    for model_name, model in models.items():
        st.subheader(f"{model_name} Evaluation (Original Data)")
        st.write(f"Training Model: {model_name}")
        evaluate_model_multiclass(model, X_train, y_train, X_test, y_test)

    # Model Evaluation on undersampled data
    for model_name, model in models.items():
        st.subheader(f"{model_name} Evaluation (Undersampled Data)")
        st.write(f"Training Model: {model_name}")
        evaluate_model_multiclass(model, X_train_under, y_train_under, X_test, y_test)

    # Neural Network Model (ANN)
    st.subheader("Neural Network (ANN) Evaluation")
    ann = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ann.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)

    y_pred_ann = (ann.predict(X_test) > 0.5).astype(int)
    st.write("Classification Report for ANN:")
    st.text(classification_report(y_test, y_pred_ann))

# # Tab: Conclusion
# elif selected_tab == "Conclusion":
#     st.header("Conclusion")
#     st.write("""
#     Summarize the key findings of the project here.
#     """)

# # Tab: Future Scope
# elif selected_tab == "Future Scope":
#     st.header("Future Scope")
#     st.write("""
#     Discuss the possible improvements and extensions for the project here.
#     Example: Using deep learning models for better accuracy.
#     """)
