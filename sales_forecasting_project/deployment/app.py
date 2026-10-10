from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_model.joblib"

st.set_page_config(page_title="Sales Revenue Predictor", page_icon="📈", layout="centered")
st.title("Product-Store Sales Revenue Predictor")
st.write(
    "Enter product and store characteristics to estimate product-store sales revenue. "
    "This baseline predicts revenue for one product-store record; it is not a date-based forecast."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("The trained model is not present yet. Run the GitHub Actions workflow first.")
    st.stop()
model = load_model()

with st.form("sales_inputs"):
    st.subheader("Product details")
    product_weight = st.number_input("Product weight", min_value=0.0, value=12.0)
    sugar_content = st.selectbox("Product sugar content", ["Low Sugar", "Regular", "No Sugar"])
    allocated_area = st.number_input("Product allocated area ratio", min_value=0.0, value=0.05, format="%.4f")
    product_type = st.selectbox(
        "Product type",
        ["Meat", "Snack Foods", "Hard Drinks", "Dairy", "Canned", "Soft Drinks",
         "Health and Hygiene", "Baking Goods", "Breads", "Breakfast", "Frozen Foods",
         "Fruits and Vegetables", "Household", "Seafood", "Starchy Foods", "Others"],
    )
    product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)

    st.subheader("Store details")
    establishment_year = st.number_input("Store establishment year", min_value=1900, max_value=2100, value=2010, step=1)
    store_size = st.selectbox("Store size", ["Small", "Medium", "High"])
    city_type = st.selectbox("Store location city type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox(
        "Store type",
        ["Departmental Store", "Supermarket Type 1", "Supermarket Type 2", "Food Mart"],
    )
    submitted = st.form_submit_button("Predict sales revenue")

if submitted:
    input_df = pd.DataFrame([{
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar_content,
        "Product_Allocated_Area": allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Establishment_Year": establishment_year,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
    }])
    try:
        predicted_revenue = float(model.predict(input_df)[0])
        st.success("Prediction completed")
        st.metric("Predicted product-store revenue", f"{predicted_revenue:,.2f}")
        with st.expander("View submitted inputs"):
            st.dataframe(input_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.caption("Check that the app's category labels match the values in the training CSV.")
