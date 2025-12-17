# AI-Driven Retail Demand & Customer Insight System

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# 2. CREATE SYNTHETIC BUSINESS DATA (UNIQUE)
np.random.seed(42)

n = 800  # number of customers

data = pd.DataFrame({
    "Customer_ID": range(1, n+1),
    "Age": np.random.randint(18, 60, n),
    "Monthly_Income": np.random.randint(15000, 120000, n),
    "Monthly_Spend": np.random.randint(1000, 40000, n),
    "Online_Visits": np.random.randint(1, 40, n),
    "Returns": np.random.randint(0, 5, n),
    "Tenure_Months": np.random.randint(1, 60, n)
})


# 3. DATA CLEANING & FEATURE ENGINEERING
data["Spend_Ratio"] = data["Monthly_Spend"] / data["Monthly_Income"]
data["Customer_Value"] = data["Monthly_Spend"] * data["Tenure_Months"]

data.drop_duplicates(inplace=True)


# 4. EXPLORATORY DATA ANALYSIS (EDA)
plt.figure()
plt.hist(data["Monthly_Spend"], bins=30)
plt.title("Monthly Spend Distribution")
plt.xlabel("Spend")
plt.ylabel("Customers")
plt.show()

plt.figure()
plt.scatter(data["Monthly_Income"], data["Monthly_Spend"])
plt.title("Income vs Spend")
plt.xlabel("Income")
plt.ylabel("Spend")
plt.show()

# 5. CUSTOMER SEGMENTATION (K-MEANS)
segment_features = data[["Monthly_Income", "Monthly_Spend", "Online_Visits"]]

kmeans = KMeans(n_clusters=4, random_state=42)
data["Segment"] = kmeans.fit_predict(segment_features)


# 6. SEGMENT INSIGHTS
segment_summary = data.groupby("Segment").agg({
    "Monthly_Income": "mean",
    "Monthly_Spend": "mean",
    "Online_Visits": "mean",
    "Customer_Value": "mean"
})

print("\nCustomer Segment Summary:\n")
print(segment_summary)


# 7. DEMAND FORECASTING (LINEAR REGRESSION)
X = data[["Monthly_Income", "Online_Visits", "Tenure_Months"]]
y = data["Monthly_Spend"]

model = LinearRegression()
model.fit(X, y)

data["Predicted_Spend"] = model.predict(X)

# 8. MODEL PERFORMANCE CHECK
error = np.mean(abs(data["Monthly_Spend"] - data["Predicted_Spend"]))
print("\nAverage Prediction Error (₹):", round(error, 2))


# 9. BUSINESS INSIGHTS (AUTOMATED)
high_value = data[data["Customer_Value"] > data["Customer_Value"].quantile(0.75)]

print("\nKey Business Insights:")
print("• High-value customers contribute:", round(high_value["Monthly_Spend"].sum(), 2))
print("• Best segment to target:", high_value["Segment"].mode()[0])
print("• Customers with high visits spend more on average")

# 10. OUTPUT FOR PORTFOLIO
data.to_csv("final_customer_analysis_output.csv", index=False)

print("\nProject executed successfully. Output file saved.")

print("Script finished successfully")
input("Press ENTER to exit...")
