import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from imblearn.over_sampling import SMOTE

# Ignore warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load cleaned dataset
benign_df = pd.read_csv("clean_benign_data.csv")
ransomware_df = pd.read_csv("clean_ransomware_data.csv")

# Combine the two datasets
df = pd.concat([benign_df, ransomware_df], ignore_index=True)

# Split into features and label
X = df.drop(columns=["label"])
y = df["label"]

# Feature Engineering - Log transformation for skewed features
skewed_features = ['file_size', 'avg_cpu_usage', 'avg_memory_usage', 'file_entropy', 'num_network_connections']
for feature in skewed_features:
    if feature in X.columns:
        X[f'{feature}_log'] = np.log1p(X[feature].clip(lower=0))

# Feature Engineering - Ratios and flags
if 'avg_cpu_usage' in X.columns and 'avg_memory_usage' in X.columns:
    X['cpu_mem_ratio'] = X['avg_cpu_usage'] / (X['avg_memory_usage'] + 1e-5)

# Replace any remaining inf/nan just in case
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Best parameters from previous optimization
best_params = {
    'bootstrap': False,
    'class_weight': 'balanced_subsample',
    'max_depth': 50,
    'max_features': 'log2',
    'min_samples_leaf': 1,
    'min_samples_split': 3,
    'n_estimators': 226
}

# Train best model directly
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train_bal, y_train_bal)

# Predictions with chosen threshold
y_proba = best_model.predict_proba(X_test)[:, 1]
best_threshold = 0.56

# Final predictions using best threshold
y_pred_custom = (y_proba >= best_threshold).astype(int)

# Evaluation
precision = precision_score(y_test, y_pred_custom)

# Plot customization
dark_red = "#D72638"
dark_bg = "#1A1A1A"
light_gray = "#EEEEEE"

sns.set(style="whitegrid")
plt.style.use('dark_background')

fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=dark_bg, gridspec_kw={'width_ratios': [2, 1]})
fig.suptitle("Random Forest Evaluation Dashboard", fontsize=18, color=light_gray, weight='bold')

# Feature Importances
importances = best_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
sns.barplot(x="Importance", y="Feature", data=importance_df.head(15), palette='crest', ax=axes[0])
axes[0].set_title("Top 15 Feature Importances", fontsize=14, color=light_gray)
axes[0].tick_params(colors=light_gray)
axes[0].set_xlabel("Importance", color=light_gray)
axes[0].set_ylabel("Feature", color=light_gray)

# Metrics Display
axes[1].axis("off")
metrics_text = (
    f"Precision: {precision * 100:.2f}%\n"
)
axes[1].text(0.5, 0.5, metrics_text, fontsize=16, ha='center', va='center', color=light_gray, family='monospace', bbox=dict(facecolor=dark_red, alpha=0.2, boxstyle='round,pad=1.5'))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()