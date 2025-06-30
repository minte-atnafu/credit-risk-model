Credit Scoring Model - Bati Bank x eCommerce Platform
🔍 Project Overview
This project aims to build a Credit Scoring System that enables Bati Bank to offer a Buy Now, Pay Later (BNPL) service in collaboration with an emerging eCommerce partner. The core objective is to predict a customer's creditworthiness based on their behavioral and transactional data, enabling automated, data-driven loan decisions that align with the Basel II Capital Accord risk management framework.

🧠 What is Credit Scoring?
Credit Scoring is the practice of assigning a numerical score to an individual that estimates the likelihood they will default on a loan. This score informs lenders whether to approve or reject a loan application and under what terms.

In this project, we aim to:

Quantify credit risk based on behavioral data.

Use machine learning to model default probabilities.

Translate risk scores into actionable credit limits and repayment durations.

⚙️ Project Pipeline
1. Define a Proxy Variable for Credit Risk
Since direct information on default may be limited, we define a proxy variable that approximates default behavior.
For instance:

Users who fail to make payments within a certain grace period.

Users with negative balances or high refund/return rates.

This binary proxy (0 = Good / Low Risk, 1 = Bad / High Risk) serves as the target label for training.

2. Feature Engineering with RFM Analysis
We extract meaningful features using Recency, Frequency, and Monetary (RFM) behavior metrics:

Recency: How recently a user made a purchase.

Frequency: How often they buy.

Monetary: How much they typically spend.

These features are supplemented by:

Product categories purchased

Device types used

Geolocation data

Time-of-day activity

Return/refund behavior

3. Train Risk Probability Model
We train a classification model (e.g., Logistic Regression, XGBoost, or Random Forest) to output a risk probability score:
P(default) ∈ [0, 1]
This probability expresses the likelihood of a customer defaulting on a credit purchase.

4. Assign Credit Score
Based on the risk probability, we map users into a credit score range (e.g., 300–850):

Low risk → High credit score

High risk → Low credit score

The mapping function can follow a non-linear transformation to reflect risk sensitivity (e.g., logistic or exponential scale).

5. Predict Optimal Loan Amount & Duration
Using regression techniques, we predict:

Maximum loan amount based on customer profile and spending habits.

Recommended repayment duration based on affordability, frequency of purchases, and risk group.

📘 Basel II Compliance
The model aligns with Basel II Capital Accord requirements by:

Estimating Probability of Default (PD) using real transaction data.

Informing decisions about Exposure at Default (EAD) and Loss Given Default (LGD) indirectly.

Providing transparency in risk estimation for regulators and internal audit.

Basel II requires institutions to have:

A risk quantification model with performance monitoring

Data-backed justifications for credit decisions

Segmentation of customers based on risk profiles

This project forms the analytical foundation to satisfy these requirements.