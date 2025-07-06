import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# CSV file path
csv_file = r'C:\Users\RAVURI VANUSSHKA\Downloads\updated_donor_data.csv'

# Load or initialize data
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame(columns=[
        "Donor Name", "Email ID", "Phone Number", "Blood Group", "City", "Hospital Name",
        "No. of Donations", "Months Since Last Donation", "Email Response Time", "Available"
    ])

# Normalize columns
df["City"] = df["City"].astype(str).str.strip().str.title()
df["Blood Group"] = df["Blood Group"].astype(str).str.strip().str.upper()
df["No. of Donations"] = pd.to_numeric(df["No. of Donations"], errors="coerce").fillna(0)
df["Months Since Last Donation"] = pd.to_numeric(df["Months Since Last Donation"], errors="coerce").fillna(0)
df["Email Response Time"] = pd.to_numeric(df["Email Response Time"], errors="coerce").fillna(24)

# Ensure 'Available' column exists
if "Available" not in df.columns:
    df["Available"] = None

# Train ML model
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
st.markdown("### Donor availability prediction system")

menu = st.sidebar.radio("Choose an action", ["Register as Donor", "Search for Donor"])

# Register as Donor
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
        response = response if response > 0 else 24

        new_row = pd.DataFrame([[
            name.strip(), email.strip(), phone.strip(), blood.strip().upper(), city.strip().title(),
            hospital.strip(), donations, months, response, None
        ]], columns=df.columns)

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_file, index=False)
        st.success("✅ Donor registered successfully!")

# Search for Donor
elif menu == "Search for Donor":
    st.subheader("🔍 Search for Donors")

    if df.empty:
        st.warning("No donor data available.")
    else:
        blood_filter = st.selectbox("Select Blood Group", sorted(df["Blood Group"].dropna().unique()))
        city_filter = st.text_input("Enter City")

        filtered = df[
            (df["Blood Group"] == blood_filter) &
            (df["City"].str.lower() == city_filter.strip().lower())
        ].copy()

        if filtered.empty:
            st.warning("No matching donors found.")
        else:
            if model and scaler:
                X = filtered[["No. of Donations", "Months Since Last Donation", "Email Response Time"]]
                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)
                filtered["Available"] = ["Yes" if p == 1 else "No" for p in preds]
            else:
                filtered["Available"] = "Unknown"

            st.write("✅ Matching Donors:")
            st.dataframe(filtered[[
                "Donor Name", "Phone Number", "Email ID", "Blood Group", "City", "Hospital Name", "Available"
            ]].reset_index(drop=True))
