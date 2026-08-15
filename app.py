import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# PAGE SETTINGS

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telco Customer Churn Prediction")
st.write(
    "Enter customer details below to predict whether the customer "
    "is likely to churn."
)

# LOAD DATASET

@st.cache_data
def load_data():
    file_path = "WA_Fn-UseC_-Telco-Customer-Churn.xlsx"
    df = pd.read_excel(file_path)

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    return df


df = load_data()

target = "Churn"

X = df.drop(columns=[target, "customerID"], errors="ignore")
y = df[target].map({"Yes": 1, "No": 0})

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# MODEL STATUS

st.success("Machine Learning Model Loaded Successfully!")

st.info(
    f"Model accuracy on the test data: {accuracy:.2%}"
)



st.header(" Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col2:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )



st.header(" Services")

col3, col4 = st.columns(2)

with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

with col4:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


st.header(" Financial Information")

col5, col6 = st.columns(2)

with col5:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

with col6:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=monthly_charges * tenure
    )



st.divider()

if st.button("Predict Customer Churn", use_container_width=True):

    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    prediction = model.predict(customer_data)[0]

    probability = model.predict_proba(customer_data)[0][1]

   

    st.header(" Prediction Result")

    if prediction == 1:
        st.error(
            f"Customer is likely to CHURN "
            f"({probability:.1%} probability)"
        )
    else:
        st.success(
            f"✅ Customer is likely to STAY "
            f"({1 - probability:.1%} probability)"
        )


st.header("🔍 How Customer Churn Prediction Works")

st.write(
    """
    This Machine Learning application predicts whether a telecom
    customer is likely to leave the company.

    The dataset contains customer information such as demographic
    details, services used, contract type, payment method, tenure,
    monthly charges and total charges.

    The customer ID is removed because it is only an identifier and
    does not help the model make a prediction.

    Categorical data is converted into numerical features using
    One-Hot Encoding. Numerical features are scaled before being
    given to the Logistic Regression model.

    The dataset is divided into training and testing data.
    The model learns patterns from the training data and predicts
    whether a new customer is likely to churn.
    """
)



st.header("📊 Dataset Information")

col7, col8, col9 = st.columns(3)

with col7:
    st.metric("Total Records", len(df))

with col8:
    st.metric("Number of Features", X.shape[1])

with col9:
    st.metric("Test Accuracy", f"{accuracy:.2%}")

st.caption("Customer Churn Prediction | Machine Learning Project")

