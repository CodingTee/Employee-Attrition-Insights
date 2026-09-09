# Employee Attrition Analysis & Prediction (R)

A complete data analysis project that investigates **why employees leave** and builds predictive models to flag at-risk employees, implemented entirely in R as coursework for **CT127-3-2-PFDA: Programming for Data Analysis** (APU, Technology Park Malaysia).

## 📊 Project Overview

Employee attrition is costly, and most of its drivers are observable in advance. This project analyses an HR dataset of **2,000 employees × 35 attributes** through a structured *Observe → Drill-Down → Cross-Validate* methodology: every analysis answers the "WHY" raised by the previous one, rather than presenting parallel, disconnected views.

The pipeline ends with a **Random Forest risk-scoring system** that assigns every employee a 0–100 risk score and an intervention tier (Low / Medium / High / Critical) for HR.

## 📁 Repository Structure

```
.
├── Code.R                          # Full analysis pipeline (run top-to-bottom)
├── dataset_employee_attrition.csv  # Dataset: 2,000 records, 35 columns
├── Documentation .docx             # Full written report (APA referenced)
└── README.md
```

## 🔬 Analysis Structure

### Objective 1: Demographic Impact
| Analysis | Question | Method |
|---|---|---|
| 1-1 | Which age group has the highest raw attrition rate? | Bar chart + ANOVA |
| 1-2 | Within that age group, do gender & marital status explain the pattern? | Drill-down + Chi-square |
| 1-3 | Does education level compound the risk in the at-risk sub-group? | Faceted comparison vs baseline |

### Objective 2: Job & Compensation
| Analysis | Question | Method |
|---|---|---|
| 2-1 | Does monthly income differ between leavers and stayers? | Violin + box plot + Welch t-test |
| 2-2 | Do overtime and income level interact? (low-income OT = highest risk?) | Interaction chart + Chi-square + Kruskal-Wallis |
| 2-3 | Where is the problem worst? | Department bubble chart |

### Objective 3: Work Experience & Tenure
| Analysis | Question | Method |
|---|---|---|
| 3-1 | When do employees most often leave? ("danger window") | Density plot + Spearman correlation |
| 3-2 | Does lack of promotion amplify attrition inside the window? | Stacked bars + Wilcoxon rank-sum |
| 3-3 | Is "new joiner who has job-hopped before" the riskiest profile? | Triple-risk heatmap + Chi-square |

### Objective 4: Predictive Modelling
| Model | Role | Notes |
|---|---|---|
| Logistic Regression | Baseline on raw imbalanced data | Deliberately exposes the low-Recall problem |
| Random Forest | Main model, trained on SMOTE-balanced data | Fixes Recall; feature importance closes the EDA loop |
| Decision Tree | HR-actionable rules from RF top-8 features | ROC comparison quantifies the interpretability cost |

Models are evaluated with Accuracy, Precision, **Recall** (the critical metric, since missing a leaver costs more than a false alarm), F1 and AUC-ROC.

### Extra Features
1. **Correlation matrix:** multicollinearity screening and feature-selection guidance
2. **K-Means clustering** (k=3 via elbow method): unsupervised risk segments, validated against actual attrition rates
3. **Risk scoring system:** RF probabilities converted to HR intervention tiers

## 🧰 Tech Stack

- **Language:** R
- **Data wrangling:** `dplyr`, `tidyr`, `reshape2`
- **Visualisation:** `ggplot2`, `ggpubr`, `ggcorrplot`, `corrplot`, `gridExtra`, `factoextra`, `rpart.plot`, `RColorBrewer`, `scales`
- **Statistical testing:** base R stats (`aov`, `chisq.test`, `t.test`, `kruskal.test`, `wilcox.test`, `cor.test`)
- **Machine learning:** `caret`, `randomForest`, `rpart`, `ROCR`, `pROC`, `cluster`, `ROSE` (SMOTE oversampling)

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/employee-attrition-analysis.git
   cd employee-attrition-analysis
   ```
2. Open `Code.R` in **RStudio** (the script auto-sets its working directory via `rstudioapi`).
3. Run the entire script. Missing packages are **installed automatically** on first run:
   ```r
   source("Code.R", echo = TRUE)
   ```

> The script is deterministic: `set.seed(42)` is set globally, so results are reproducible.

## 📌 Key Findings

- **Demographics:** The youngest cohort (18–25) and single employees show the highest attrition; education compounds risk within the at-risk sub-group.
- **Compensation:** Leavers earn significantly less; **low income + overtime** interacts into a compounded risk profile, worst in specific departments.
- **Tenure:** A clear "danger window" exists in the **first 0–3 years**; promotion stagnation amplifies risk, and serial job-hoppers inside the window form the maximum-risk profile.
- **Modelling:** Random Forest (SMOTE-balanced) delivers the best Recall/AUC; its top features confirm the EDA findings from Objectives 1–3, closing the analysis loop.

## 📄 Documentation

The full written report, including data description, assumptions, hypothesis, per-analysis techniques, findings and interpretation, is in `Documentation .docx`.

## ⚠️ Disclaimer

This project was completed as coursework for CT127-3-2-PFDA (Programming for Data Analysis). The dataset is for educational purposes only.
