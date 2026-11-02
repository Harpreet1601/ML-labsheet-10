"""Disease project - single file: data generation, training, Streamlit app.

Usage:
  python disease_project.py          # generate data, train, save model
  streamlit run disease_project.py   # prediction interface (after training)
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
PER_DISEASE = 250

PROFILES = {
    "Common Cold": {"cough": .8, "runny_nose": .9, "sneezing": .8, "sore_throat": .6, "headache": .3, "fever": .25, "fatigue": .3},
    "Influenza": {"fever": .9, "cough": .8, "body_ache": .85, "fatigue": .85, "headache": .7, "chills": .7, "sore_throat": .4},
    "COVID-19": {"fever": .8, "cough": .85, "loss_of_smell": .7, "fatigue": .75, "breathlessness": .5, "body_ache": .5, "headache": .45, "sore_throat": .35},
    "Malaria": {"fever": .95, "chills": .9, "sweating": .85, "headache": .7, "nausea": .55, "vomiting": .45, "body_ache": .5, "fatigue": .6},
    "Dengue": {"fever": .95, "headache": .8, "joint_pain": .8, "skin_rash": .6, "body_ache": .85, "nausea": .5, "vomiting": .4, "fatigue": .65},
    "Typhoid": {"fever": .9, "abdominal_pain": .7, "headache": .6, "loss_of_appetite": .8, "fatigue": .7, "diarrhea": .4, "nausea": .4},
    "Gastroenteritis": {"diarrhea": .9, "vomiting": .8, "nausea": .85, "abdominal_pain": .85, "fever": .4, "loss_of_appetite": .6, "fatigue": .45},
    "Migraine": {"headache": .98, "nausea": .65, "vomiting": .35, "light_sensitivity": .85, "dizziness": .4},
    "Hypertension": {"headache": .55, "dizziness": .55, "chest_pain": .3, "breathlessness": .3, "fatigue": .35, "blurred_vision": .3},
    "Diabetes": {"frequent_urination": .85, "excessive_thirst": .85, "fatigue": .65, "blurred_vision": .45, "weight_loss": .45, "loss_of_appetite": .15},
    "Asthma": {"breathlessness": .9, "cough": .7, "wheezing": .85, "chest_pain": .35, "fatigue": .3},
    "Pneumonia": {"fever": .85, "cough": .9, "breathlessness": .75, "chest_pain": .6, "chills": .55, "fatigue": .6, "sweating": .4},
}
SYMPTOMS = sorted({s for p in PROFILES.values() for s in p})
NOISE = 0.04
# typical (mean, sd) body temperature (C) and age (mean, sd)
TEMP = {"fever": (38.9, 0.6), "no_fever": (36.9, 0.3)}
AGE = {"Migraine": (32, 10), "Hypertension": (55, 12), "Diabetes": (52, 13),
       "Asthma": (28, 15), "Pneumonia": (45, 20)}


def build_dataset() -> pd.DataFrame:
    rows = []
    for disease, prof in PROFILES.items():
        for _ in range(PER_DISEASE):
            r = {s: int(RNG.random() < prof.get(s, NOISE)) for s in SYMPTOMS}
            m, sd = TEMP["fever" if r.get("fever") else "no_fever"]
            r["body_temp_c"] = round(float(RNG.normal(m, sd)), 1)
            am, asd = AGE.get(disease, (35, 18))
            r["age"] = int(np.clip(RNG.normal(am, asd), 1, 90))
            r["gender"] = RNG.choice(["Male", "Female"])
            r["disease"] = disease
            rows.append(r)
    df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    # a few missing temperatures, to exercise imputation
    idx = RNG.choice(len(df), int(len(df) * 0.03), replace=False)
    df.loc[idx, "body_temp_c"] = np.nan
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
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NUM = ["body_temp_c", "age"]
CAT = ["gender"]


def main() -> None:
    df = pd.read_csv("disease_data.csv")
    symptoms = [c for c in df.columns if c not in NUM + CAT + ["disease"]]
    print("Shape:", df.shape, "| symptoms:", len(symptoms))
    print("Class balance:\n", df["disease"].value_counts().head(3), "...")

    # EDA
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.groupby("disease")[symptoms].mean(), cmap="YlOrRd")
    plt.title("Symptom frequency by disease")
    plt.tight_layout()
    plt.savefig("symptom_heatmap.png", dpi=130)
    plt.close()

    le = LabelEncoder()
    y = le.fit_transform(df["disease"])
    X = df.drop(columns="disease")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("sym", "passthrough", symptoms),
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
    }

    rows, fitted = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="accuracy").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({"Model": name, "CV Accuracy": cv,
                     "Accuracy": accuracy_score(y_te, pred),
                     "Precision": precision_score(y_te, pred, average="weighted", zero_division=0),
                     "Recall": recall_score(y_te, pred, average="weighted"),
                     "F1": f1_score(y_te, pred, average="weighted")})
        fitted[name] = pipe

    results = pd.DataFrame(rows).set_index("Model").round(4)
    print("\nModel comparison:\n", results)
    results.to_csv("model_comparison.csv")

    best = results["Accuracy"].idxmax()
    final = fitted[best]
    pred = final.predict(X_te)
    print(f"\nBest model: {best}\n")
    print(classification_report(y_te, pred, target_names=le.classes_))

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=le.classes_,
                                            xticks_rotation=60, ax=ax, colorbar=False)
    plt.title(f"Confusion matrix - {best}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=130)
    plt.close()

    results[["Accuracy", "F1"]].plot(kind="bar", figsize=(9, 5), ylim=(0.5, 1))
    plt.title("Model comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=130)
    plt.close()

    joblib.dump({"model": final, "label_encoder": le, "symptoms": symptoms,
                 "best_name": best, "accuracy": float(accuracy_score(y_te, pred))},
                "disease_model.joblib")
    print("Saved disease_model.joblib")




def run_app():
    import streamlit as st

    st.set_page_config(page_title="Disease Predictor", page_icon="🩺")
    st.title("🩺 Disease Prediction System")
    st.caption("MCA III Sem - Capstone ML Project. Educational demo only, not medical advice.")


    @st.cache_resource
    def load_bundle():
        return joblib.load("disease_model.joblib")


    bundle = load_bundle()
    model, le, symptoms = bundle["model"], bundle["label_encoder"], bundle["symptoms"]
    st.sidebar.write(f"Model: **{bundle['best_name']}**")
    st.sidebar.write(f"Test accuracy: **{bundle['accuracy']:.1%}**")

    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 1, 100, 30)
    gender = c2.selectbox("Gender", ["Male", "Female"])
    temp = c3.number_input("Body temperature (°C)", 34.0, 42.0, 36.9, 0.1)

    chosen = st.multiselect("Select symptoms", symptoms,
                            format_func=lambda s: s.replace("_", " ").title())

    if st.button("Predict", type="primary"):
        if not chosen:
            st.warning("Please select at least one symptom.")
        else:
            row = {s: int(s in chosen) for s in symptoms}
            row.update({"body_temp_c": temp, "age": age, "gender": gender})
            proba = model.predict_proba(pd.DataFrame([row]))[0]
            top = proba.argsort()[::-1][:3]
            st.subheader(f"Most likely: {le.classes_[top[0]]} ({proba[top[0]]:.0%})")
            st.table(pd.DataFrame({"Disease": le.classes_[top],
                                   "Probability": [f"{proba[i]:.1%}" for i in top]}))
            st.info("Consult a qualified doctor for any real medical concern.")


CSV_NAME = "disease_data.csv"


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