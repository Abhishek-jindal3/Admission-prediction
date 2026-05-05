

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("Admission_Predict.csv")
df.columns = df.columns.str.strip()

if "Serial No." in df.columns:
    df = df.drop("Serial No.", axis=1)


X = df.drop("Chance of Admit", axis=1)
y = df["Chance of Admit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nMODEL PERFORMANCE")
print("R2 Score:", round(r2_score(y_test, pred), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 3))


importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

sns.set_style("whitegrid")
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")

plt.title("Feature Importance", fontsize=15, weight="bold")

for i, v in enumerate(importance_df["Importance"]):
    plt.text(v, i, f"{v:.2f}")

plt.show()


def analyze_profile(model, student):
    df_input = pd.DataFrame([student])
    score = model.predict(df_input)[0]

    print("\n🎓 PROFILE ANALYSIS")
    print("-"*40)
    for k, v in student.items():
        print(f"{k}: {v}")

    print("\n Current Admission Chance:", round(score, 3))

    if score > 0.8:
        print(" Strong profile")
    elif score > 0.6:
        print(" Decent but improvable")
    else:
        print(" Weak profile")

    return score


def strategy_optimizer(model, student):
    base_score = model.predict(pd.DataFrame([student]))[0]

    print("\n OPTIMAL IMPROVEMENT STRATEGY")
    print("-"*40)

    improvements = []

    for feature in student:
        temp = student.copy()


        if feature == "CGPA":
            temp[feature] += 0.5
        elif feature in ["GRE Score", "TOEFL Score"]:
            temp[feature] += 5
        elif feature in ["SOP", "LOR", "University Rating"]:
            temp[feature] += 1
        elif feature == "Research":
            temp[feature] = 1

        new_score = model.predict(pd.DataFrame([temp]))[0]
        gain = new_score - base_score

        improvements.append((feature, gain))

    # sort by best improvement
    improvements.sort(key=lambda x: x[1], reverse=True)

    total_gain = 0

    for feature, gain in improvements:
        if gain > 0:
            print(f" Improve {feature} → +{gain:.3f}")
            total_gain += gain

    print("\n Total Possible Improvement:", round(total_gain, 3))
    print(" Optimized Chance:", round(base_score + total_gain, 3))


student = {
    "GRE Score": 310,
    "TOEFL Score": 105,
    "University Rating": 3,
    "SOP": 3,
    "LOR": 3,
    "CGPA": 8.0,
    "Research": 0
}


base = analyze_profile(model, student)
strategy_optimizer(model, student)
