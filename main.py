import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration ---
st.set_page_config(page_title="BrainWaveAI", page_icon="🧠")

st.title("🧠 BrainWaveAI")
st.write("### Intelligent Predictive Analytics Tool")
st.write("Upload a dataset, select your target, and let AI do the rest.")

# --- 1. Data Ingestion ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # --- 2. Configuration ---
    st.sidebar.header("Model Configuration")
    
    # Select Target Variable
    target_column = st.sidebar.selectbox("Select Column to Predict (Target)", df.columns)
    
    # Select Features (Drop target and non-numeric columns automatically for simplicity)
    feature_columns = st.sidebar.multiselect("Select Features (Inputs)", 
                                             [c for c in df.columns if c != target_column],
                                             default=[c for c in df.columns if c != target_column])
    
    if st.sidebar.button("Train BrainWave Model"):
        if not feature_columns:
            st.error("Please select at least one feature.")
        else:
            # --- 3. Preprocessing ---
            X = df[feature_columns]
            y = df[target_column]
            
            # Handle Non-Numeric Data (Simple Encoding)
            X = pd.get_dummies(X) # One-hot encoding for inputs
            
            is_classification = False
            
            # Check if target is categorical (Text or few unique numbers)
            if y.dtype == 'object' or y.nunique() < 10:
                is_classification = True
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # --- 4. Model Training ---
            st.write("---")
            st.write("### ⚙️ Training Model...")
            
            if is_classification:
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                st.success(f"Model Trained! (Classification Mode)")
                st.metric("Accuracy", f"{acc * 100:.2f}%")
            else:
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                st.success(f"Model Trained! (Regression Mode)")
                st.metric("Mean Squared Error", f"{mse:.4f}")
            
            # --- 5. Prediction Interface ---
            st.write("### 🔮 Make a Prediction")
            st.write("Enter values for a new prediction:")
            
            input_data = {}
            for col in feature_columns:
                # Create input fields based on column type in original dataframe
                if df[col].dtype == 'object':
                    val = st.selectbox(f"{col}", df[col].unique())
                else:
                    val = st.number_input(f"{col}", value=0.0)
                input_data[col] = val
            
            if st.button("Predict Result"):
                # Align input data with training structure (get_dummies alignment)
                input_df = pd.DataFrame([input_data])
                input_df = pd.get_dummies(input_df)
                input_df = input_df.reindex(columns=X.columns, fill_value=0)
                
                prediction = model.predict(input_df)
                
                if is_classification:
                    final_result = le.inverse_transform(prediction)[0]
                    st.success(f"**Predicted {target_column}: {final_result}**")
                else:
                    st.success(f"**Predicted {target_column}: {prediction[0]:.2f}**")

else:
    st.info("Awaiting CSV file upload to start.")