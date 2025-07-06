import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# CSV file path
csv_file = r'C:\Users\RAVURI VANUSSHKA\Downloads\final_data_converted.csv'

# Load or initialize data
if os.path.exists(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        df = pd.DataFrame()
else:
    df = pd.DataFrame(columns=[
        "Donor Name", "Email ID", "Phone Number", "Blood Group", "City",
        "Hospital Name", "No. of Donations", "Months Since Last Donation",
        "Email Response Time", "Available"
    ])

# Ensure all columns exist
required_cols = [
    "Donor Name", "Email ID", "Phone Number", "Blood Group", "City",
    "Hospital Name", "No. of Donations", "Months Since Last Donation",
    "Email Response Time", "Available"
]
for col in required_cols:
    if col not in df.columns:
        df[col] = None

# Normalize data
df["City"] = df["City"].astype(str).str.strip().str.title()
df["Blood Group"] = df["Blood Group"].astype(str).str.strip().str.upper()
df["No. of Donations"] = pd.to_numeric(df["No. of Donations"], errors='coerce').fillna(0)
df["Months Since Last Donation"] = pd.to_numeric(df["Months Since Last Donation"], errors='coerce').fillna(0)
df["Email Response Time"] = pd.to_numeric(df["Email Response Time"], errors='coerce').fillna(24)

# Train logistic regression model for availability prediction
def train_model(data):
    X = data[["No. of Donations", "Months Since Last Donation", "Email Response Time"]]
    y = [
        1 if d > 0 and m <= 6 and r <= 24 else 0
        for d, m, r in zip(X["No. of Donations"], X["Months Since Last Donation"], X["Email Response Time"])
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = None, None
if not df.empty and len(df) >= 5:
    model, scaler = train_model(df)

# Streamlit UI
st.title("🩸 DonorConnect")
st.markdown("### Donor Availability Prediction System")

menu = st.sidebar.radio("Choose an action", ["Register as Donor", "Search for Donor"])

# Donor Registration Section
if menu == "Register as Donor":
    st.subheader("📋 Donor Registration")

    name = st.text_input("Donor Name")
    email = st.text_input("Email ID")
    phone = st.text_input("Phone Number")
    blood = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    city = st.text_input("City")
    hospital = st.text_input("Hospital Name")
    donations = st.number_input("No. of Donations", min_value=0, step=1)
    months = st.number_input("Months Since Last Donation", min_value=0, step=1)
    response = st.number_input("Email Response Time (in hours)", min_value=0, step=1)

    if st.button("Register"):
        new_row = pd.DataFrame([[
            name.strip(),
            email.strip(),
            phone.strip(),
            blood.strip().upper(),
            city.strip().title(),
            hospital.strip(),
            donations,
            months,
            response if response > 0 else 24,
            None
        ]], columns=df.columns)

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_file, index=False)
        st.success("✅ Donor registered successfully!")

# Donor Search Section with dropdown blood group and text input city
elif menu == "Search for Donor":
    st.subheader("🔍 Search for Donors")

    if df.empty:
        st.warning("⚠️ No donor data available.")
    else:
        # Prepare blood groups for dropdown (unique or default list)
        all_blood_groups = sorted(df["Blood Group"].dropna().unique())
        if not all_blood_groups:
            all_blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

        blood_filter = st.selectbox(
            "Select Blood Group",
            options=all_blood_groups,
            index=0,
            help="Choose blood group to filter donors"
        )

        city_filter = st.text_input("Enter City").strip()

        filtered = df[
            (df["Blood Group"] == blood_filter) &
            (df["City"].str.lower() == city_filter.lower())
        ].copy()

        if filtered.empty:
            st.info("❌ No matching donors found.")
        else:
            if model and scaler:
                X = filtered[["No. of Donations", "Months Since Last Donation", "Email Response Time"]]
                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)
                filtered["Available"] = ["Yes" if p == 1 else "No" for p in preds]
            else:
                filtered["Available"] = "Unknown"

            st.success("✅ Matching donors found:")
            st.dataframe(filtered[[
                "Donor Name", "Phone Number", "Email ID", "Blood Group",
                "City", "Hospital Name", "Available"
            ]].reset_index(drop=True))
