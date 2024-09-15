import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import streamlit as st

# Load the model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# List of feature names based on one-hot encoding
feature_names = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Female', 'gender_Male',
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Streamlit UI to collect user input
st.title("Customer Churn Prediction App")

# Collect inputs with unique keys and arrange horizontally
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure", min_value=0, max_value=100, value=0)
    monthly_charges = st.slider("Monthly Charges", min_value=0.0, max_value=200.0, value=0.0)
    total_charges = st.slider("Total Charges", min_value=0.0, max_value=10000.0, value=0.0)

with col2:
    gender = st.radio("Gender", ["Female", "Male"])
    partner = st.checkbox("Partner: Yes")
    dependents = st.checkbox("Dependents: Yes")
    phone_service = st.checkbox("Phone Service: Yes")

with col3:
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "No", "DSL"])
    online_security = st.selectbox("Online Security", ["No internet service", "Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["No internet service", "Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["No internet service", "Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["No internet service", "Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["No internet service", "Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "Yes", "No"])
    contract = st.selectbox("Contract", ["One year", "Two year", "Month-to-month"])
    paperless_billing = st.checkbox("Paperless Billing: Yes")
    payment_method = st.selectbox("Payment Method", ["Credit card (automatic)", "Electronic check", "Mailed check"])

# Process the input data
input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],

    'gender_Female': [1 if gender == "Female" else 0],
    'gender_Male': [1 if gender == "Male" else 0],

    'Partner_Yes': [1 if partner else 0],
    'Dependents_Yes': [1 if dependents else 0],
    'PhoneService_Yes': [1 if phone_service else 0],
    'MultipleLines_No phone service': [1 if multiple_lines == "No phone service" else 0],
    'MultipleLines_Yes': [1 if multiple_lines == "Yes" else 0],
    'InternetService_Fiber optic': [1 if internet_service == "Fiber optic" else 0],
    'InternetService_No': [1 if internet_service == "No" else 0],
    'OnlineSecurity_No internet service': [1 if online_security == "No internet service" else 0],
    'OnlineSecurity_Yes': [1 if online_security == "Yes" else 0],
    'OnlineBackup_No internet service': [1 if online_backup == "No internet service" else 0],
    'OnlineBackup_Yes': [1 if online_backup == "Yes" else 0],
    'DeviceProtection_No internet service': [1 if device_protection == "No internet service" else 0],
    'DeviceProtection_Yes': [1 if device_protection == "Yes" else 0],
    'TechSupport_No internet service': [1 if tech_support == "No internet service" else 0],
    'TechSupport_Yes': [1 if tech_support == "Yes" else 0],
    'StreamingTV_No internet service': [1 if streaming_tv == "No internet service" else 0],
    'StreamingTV_Yes': [1 if streaming_tv == "Yes" else 0],
    'StreamingMovies_No internet service': [1 if streaming_movies == "No internet service" else 0],
    'StreamingMovies_Yes': [1 if streaming_movies == "Yes" else 0],
    'Contract_One year': [1 if contract == "One year" else 0],
    'Contract_Two year': [1 if contract == "Two year" else 0],
    'PaperlessBilling_Yes': [1 if paperless_billing else 0],
    'PaymentMethod_Credit card (automatic)': [1 if payment_method == "Credit card (automatic)" else 0],
    'PaymentMethod_Electronic check': [1 if payment_method == "Electronic check" else 0],
    'PaymentMethod_Mailed check': [1 if payment_method == "Mailed check" else 0]
})

# Ensure input_data has the same number of features as the training data
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# Process input data
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_data_scaled_numeric = scaler.transform(input_data[numeric_cols])
input_data_other = input_data.drop(columns=numeric_cols)
input_data_scaled = np.concatenate([input_data_scaled_numeric, input_data_other], axis=1)

if st.button('Predict'):
    try:
        prediction = model.predict(input_data_scaled)
        if prediction[0] == 1:
            st.write("The customer is likely to churn.")
        else:
            st.write("The customer is not likely to churn.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")


# Function to plot EDA
def plot_eda():
    # Load the dataset
    df = pd.read_csv('Datasets.csv')

    # Data Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    # Display important points
    st.write("### Important Points for EDA:")
    st.write("1. **Total Number of Rows and Columns**: This tells us the size of the dataset.")
    st.write("2. **Missing Values**: Check how many missing values are present in each column.")
    st.write("3. **Data Types**: Understand the data types of each column.")
    st.write("4. **Summary Statistics**: Get a summary of statistics for numeric features.")
    st.write("5. **Correlation Matrix**: Visualize correlations between numeric features.")

    # Display mean values
    st.write(f"Mean Tenure: {df['tenure'].mean():.2f}")
    st.write(f"Mean Monthly Charges: {df['MonthlyCharges'].mean():.2f}")
    st.write(f"Mean Total Charges: {df['TotalCharges'].mean():.2f}")

    # Plot and save histogram for 'TotalCharges'
    fig, ax = plt.subplots()
    sns.histplot(df['TotalCharges'], bins=30, kde=True, ax=ax, color='skyblue')
    ax.set_title('Distribution of Total Charges')
    ax.set_xlabel('Total Charges')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    hist_img = io.BytesIO()
    plt.savefig(hist_img, format='png')
    plt.close()
    hist_img.seek(0)

    # Display histogram and add download button
    st.image(hist_img, caption='Distribution of Total Charges', use_column_width=True)
    st.download_button(
        label="Download Histogram of Total Charges",
        data=hist_img,
        file_name="total_charges_histogram.png",
        mime="image/png"
    )

    # Plot and save boxplot for 'MonthlyCharges' vs. 'Churn'
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=ax, palette='Set2')
    ax.set_title('Boxplot of Monthly Charges by Churn')
    ax.set_xlabel('Churn')
    ax.set_ylabel('Monthly Charges')
    plt.tight_layout()
    boxplot_img = io.BytesIO()
    plt.savefig(boxplot_img, format='png')
    plt.close()
    boxplot_img.seek(0)

    # Display boxplot and add download button
    st.image(boxplot_img, caption='Boxplot of Monthly Charges by Churn', use_column_width=True)
    st.download_button(
        label="Download Boxplot of Monthly Charges by Churn",
        data=boxplot_img,
        file_name="monthly_charges_boxplot.png",
        mime="image/png"
    )

    # Plot and save line plot for mean MonthlyCharges over tenure
    df_grouped = df.groupby('tenure')['MonthlyCharges'].mean().reset_index()
    fig, ax = plt.subplots()
    ax.plot(df_grouped['tenure'], df_grouped['MonthlyCharges'], marker='o', linestyle='-', color='orange')
    ax.set_title('Mean Monthly Charges over Tenure')
    ax.set_xlabel('Tenure')
    ax.set_ylabel('Mean Monthly Charges')
    plt.tight_layout()
    lineplot_img = io.BytesIO()
    plt.savefig(lineplot_img, format='png')
    plt.close()
    lineplot_img.seek(0)

    # Display line plot and add download button
    st.image(lineplot_img, caption='Line Plot of Mean Monthly Charges over Tenure', use_column_width=True)
    st.download_button(
        label="Download Line Plot of Mean Monthly Charges over Tenure",
        data=lineplot_img,
        file_name="monthly_charges_lineplot.png",
        mime="image/png"
    )

    # Plot and save line plot for Monthly Charges by Contract Type
    df_contract = df.groupby('Contract')['MonthlyCharges'].mean().reset_index()
    fig, ax = plt.subplots()
    ax.plot(df_contract['Contract'], df_contract['MonthlyCharges'], marker='o', linestyle='-', color='green')
    ax.set_title('Monthly Charges by Contract Type')
    ax.set_xlabel('Contract Type')
    ax.set_ylabel('Monthly Charges')
    plt.tight_layout()
    contract_lineplot_img = io.BytesIO()
    plt.savefig(contract_lineplot_img, format='png')
    plt.close()
    contract_lineplot_img.seek(0)

    # Display line plot and add download button
    st.image(contract_lineplot_img, caption='Line Plot of Monthly Charges by Contract Type', use_column_width=True)
    st.download_button(
        label="Download Line Plot of Monthly Charges by Contract Type",
        data=contract_lineplot_img,
        file_name="monthly_charges_contract_lineplot.png",
        mime="image/png"
    )

    # Plot and save line plot for Total Charges by Tenure
    df_tenure = df.groupby('tenure')['TotalCharges'].mean().reset_index()
    fig, ax = plt.subplots()
    ax.plot(df_tenure['tenure'], df_tenure['TotalCharges'], marker='o', linestyle='-', color='blue')
    ax.set_title('Total Charges by Tenure')
    ax.set_xlabel('Tenure')
    ax.set_ylabel('Total Charges')
    plt.tight_layout()
    tenure_lineplot_img = io.BytesIO()
    plt.savefig(tenure_lineplot_img, format='png')
    plt.close()
    tenure_lineplot_img.seek(0)

    # Display line plot and add download button
    st.image(tenure_lineplot_img, caption='Line Plot of Total Charges by Tenure', use_column_width=True)
    st.download_button(
        label="Download Line Plot of Total Charges by Tenure",
        data=tenure_lineplot_img,
        file_name="total_charges_tenure_lineplot.png",
        mime="image/png"
    )


# Add button to generate EDA
if st.button('Generate EDA'):
    with st.spinner('Generating EDA...'):
        plot_eda()
        st.success('EDA generated successfully!')
