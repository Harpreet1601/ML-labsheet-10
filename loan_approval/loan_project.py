"""Loan project - single file: data generation, training, Streamlit app.

Usage:
  python loan_project.py          # generate data, train, save model
  streamlit run loan_project.py   # prediction interface (after training)
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1500


def build_dataset(n: int = N) -> pd.DataFrame:
    gender = RNG.choice(["Male", "Female"], n, p=[0.8, 0.2])
    married = RNG.choice(["Yes", "No"], n, p=[0.65, 0.35])
    dependents = RNG.choice(["0", "1", "2", "3+"], n, p=[0.57, 0.17, 0.17, 0.09])
    education = RNG.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22])
    self_emp = RNG.choice(["Yes", "No"], n, p=[0.14, 0.86])
    property_area = RNG.choice(["Urban", "Semiurban", "Rural"], n, p=[0.33, 0.38, 0.29])

    base = RNG.lognormal(mean=8.4, sigma=0.55, size=n)
    applicant_income = np.round(base * np.where(education == "Graduate", 1.25, 0.85))
    coapplicant_income = np.where(
        (married == "Yes") & (RNG.random(n) < 0.6),
        np.round(RNG.lognormal(7.8, 0.6, n)),
        0,
    )
    total_income = applicant_income + coapplicant_income
    loan_amount = np.round(np.clip(total_income * RNG.uniform(0.02, 0.06, n), 20, 600))
    loan_term = RNG.choice([120, 180, 240, 300, 360, 480], n, p=[0.02, 0.05, 0.03, 0.05, 0.82, 0.03])
    credit_history = RNG.choice([1.0, 0.0], n, p=[0.84, 0.16])

    # Latent approval score: credit history dominates, as in real lending data.
    score = (
        3.0 * (credit_history - 0.5)
        + 0.9 * np.log(total_income / 6000)
        - 0.8 * np.log(loan_amount / 130)
        + 0.3 * (education == "Graduate")
        + 0.3 * (married == "Yes")
        + 0.35 * (property_area == "Semiurban")
        - 0.25 * (self_emp == "Yes")
        + RNG.normal(0, 0.8, n)
    )
    status = np.where(score > 0.35, "Y", "N")

    df = pd.DataFrame({
        "Loan_ID": [f"LP{100000 + i}" for i in range(n)],
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_emp,
        "ApplicantIncome": applicant_income.astype(int),
        "CoapplicantIncome": coapplicant_income.astype(int),
        "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history, "Property_Area": property_area,
        "Loan_Status": status,
    })
    # Inject missing values
    for col, frac in [("Gender", .02), ("Married", .01), ("Dependents", .025),
                      ("Self_Employed", .05), ("LoanAmount", .035),
                      ("Loan_Amount_Term", .02), ("Credit_History", .08)]:
        idx = RNG.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan
    return df




import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             classification_report)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CAT = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUM = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
       "Credit_History", "TotalIncome", "EMI", "Income_to_Loan"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering shared by training and the Streamlit app."""
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"] = df["LoanAmount"] * 1000 / df["Loan_Amount_Term"]
    df["Income_to_Loan"] = df["TotalIncome"] / df["LoanAmount"].replace(0, np.nan)
    return df


def eda(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    sns.countplot(data=df, x="Loan_Status", ax=ax[0, 0]).set_title("Target distribution")
    sns.countplot(data=df, x="Credit_History", hue="Loan_Status", ax=ax[0, 1]).set_title("Credit history vs status")
    sns.histplot(df["ApplicantIncome"], bins=40, ax=ax[1, 0]).set_title("Applicant income")
    sns.boxplot(data=df, x="Loan_Status", y="LoanAmount", ax=ax[1, 1]).set_title("Loan amount vs status")
    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=130)
    plt.close()


def main() -> None:
    df = pd.read_csv("loan_data.csv").drop(columns="Loan_ID")
    print("Shape:", df.shape)
    print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
    eda(df)

    df = add_features(df)
    X = df.drop(columns="Loan_Status")
    y = (df["Loan_Status"] == "Y").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    rows, fitted = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="accuracy").mean()
        pipe.fit(X_tr, y_tr)
        pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
        rows.append({"Model": name, "CV Accuracy": cv,
                     "Accuracy": accuracy_score(y_te, pred),
                     "Precision": precision_score(y_te, pred),
                     "Recall": recall_score(y_te, pred),
                     "F1": f1_score(y_te, pred),
                     "ROC-AUC": roc_auc_score(y_te, proba)})
        fitted[name] = pipe

    results = pd.DataFrame(rows).set_index("Model").round(4)
    print("\nModel comparison:\n", results)
    results.to_csv("model_comparison.csv")

    # Tune the best model (by F1) with a small grid search
    best_name = results["F1"].idxmax()
    print("\nBest baseline:", best_name)
    grids = {
        "Logistic Regression": {"clf__C": [0.1, 1, 10]},
        "Decision Tree": {"clf__max_depth": [3, 5, 7]},
        "Random Forest": {"clf__max_depth": [5, 10, None], "clf__n_estimators": [100, 300]},
        "Gradient Boosting": {"clf__learning_rate": [0.05, 0.1], "clf__n_estimators": [100, 200]},
    }
    gs = GridSearchCV(Pipeline([("pre", pre), ("clf", models[best_name])]),
                      grids[best_name], cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    final = gs.best_estimator_
    print("Best params:", gs.best_params_)
    pred = final.predict(X_te)
    print("\nFinal model report:\n", classification_report(y_te, pred, target_names=["Rejected", "Approved"]))

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Rejected", "Approved"])
    plt.title(f"Confusion matrix - {best_name} (tuned)")
    plt.savefig("confusion_matrix.png", dpi=130)
    plt.close()

    results[["Accuracy", "F1", "ROC-AUC"]].plot(kind="bar", figsize=(9, 5), ylim=(0.5, 1))
    plt.title("Model comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=130)
    plt.close()

    joblib.dump(final, "loan_model.joblib")
    print("Saved loan_model.joblib")




def run_app():
    import streamlit as st


    st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
    st.title("🏦 Loan Approval Prediction System")
    st.caption("MCA III Sem - Capstone ML Project")


    @st.cache_resource
    def load_model():
        return joblib.load("loan_model.joblib")


    model = load_model()

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital status (Married?)", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_emp = st.selectbox("Self employed?", ["No", "Yes"])
        area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
    with col2:
        app_income = st.number_input("Applicant income (monthly)", 0, 200000, 5000, 500)
        co_income = st.number_input("Co-applicant income (monthly)", 0, 200000, 0, 500)
        loan_amt = st.number_input("Loan amount (in thousands)", 10, 1000, 130, 10)
        term = st.selectbox("Loan term (months)", [120, 180, 240, 300, 360, 480], index=4)
        credit = st.selectbox("Credit history", ["Good (meets guidelines)", "Bad"])

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            "Gender": gender, "Married": married, "Dependents": dependents,
            "Education": education, "Self_Employed": self_emp,
            "ApplicantIncome": app_income, "CoapplicantIncome": co_income,
            "LoanAmount": loan_amt, "Loan_Amount_Term": term,
            "Credit_History": 1.0 if credit.startswith("Good") else 0.0,
            "Property_Area": area,
        }])
        row = add_features(row)
        proba = model.predict_proba(row)[0, 1]
        if proba >= 0.5:
            st.success(f"Likely APPROVED  (confidence {proba:.0%})")
        else:
            st.error(f"Likely REJECTED  (approval probability {proba:.0%})")
        st.progress(float(proba))


CSV_NAME = "loan_data.csv"


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit = get_script_run_ctx() is not None
    except Exception:
        in_streamlit = False
    if in_streamlit:
        run_app()
    else:
        import os
        if not os.path.exists(CSV_NAME):
            build_dataset().to_csv(CSV_NAME, index=False)
        main()