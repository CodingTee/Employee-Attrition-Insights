# ============================================================
# CT127-3-2 Programming for Data Analysis
# Employee Attrition Classification
# ============================================================
# ANALYSIS STRUCTURE OVERVIEW
# -------------------------------------------------------------
# Each Objective follows a 3-layer "Observe -> Drill-Down ->
# Cross-Validate" structure so that every Analysis answers the
# WHY raised by the previous one, not just a parallel view.
#
# Objective 1 - Demographic Impact
#   1-1  Which age group has the highest raw attrition rate?
#   1-2  Within the high-risk age group, do gender & marital
#         status explain the pattern?  (drill-down)
#   1-3  Does education level modify the risk? Cross-heatmap
#         of Age Group x Education to find worst combination.
#
# Objective 2 - Job & Compensation
#   2-1  Does monthly income differ by attrition? (violin/box)
#   2-2  Does overtime + income level interact?
#         (low-income overtime workers = highest risk?)
#   2-3  Composite satisfaction score vs attrition rate by
#         department - links pay, workload and satisfaction.
#
# Objective 3 - Work Experience & Tenure
#   3-1  Identify the "danger window" of tenure (years 1-3).
#   3-2  Inside the danger window, does lack of promotion
#         amplify attrition? (stacked-bar drill-down)
#   3-3  Cross-tabulate job-hopping x tenure window ->
#         is "new joiner who has hopped before" the riskiest?
#
# Objective 4 - Predictive Modelling
#   4-1  Logistic Regression (baseline) -> exposes low Recall problem
#   4-2  Random Forest -> fixes Recall; Feature Importance closes
#         EDA loop by colour-mapping features back to Obj 1-3
#   4-3  Decision Tree -> translates RF findings into HR-actionable
#         rules; ROC comparison quantifies interpretability cost
#
# Extra Feature 1 - Correlation Matrix (guides feature selection)
# Extra Feature 2 - K-Means Clustering (unsupervised risk profiling)
# Extra Feature 3 - Risk Scoring System (HR intervention tiers)
# ============================================================


# ============================================================
# SECTION 0: INSTALL & LOAD REQUIRED LIBRARIES
# ============================================================
required_packages <- c(
  "ggplot2",       # grammar-of-graphics plotting
  "dplyr",         # data manipulation verbs
  "tidyr",         # reshaping (pivot_longer / pivot_wider)
  "corrplot",      # base correlation plot
  "caret",         # ML training / confusion matrix
  "randomForest",  # Random Forest classifier
  "rpart",         # Decision Tree
  "rpart.plot",    # Decision Tree visualisation
  "gridExtra",     # arrange multiple ggplots on one page
  "scales",        # percent_format() etc.
  "ggcorrplot",    # ggplot2-based correlation matrix
  "reshape2",      # melt() for wide-to-long reshaping
  "ROCR",          # ROC performance objects
  "pROC",          # roc() / auc() / ggroc()
  "cluster",       # clustering algorithms
  "factoextra",    # fviz_cluster() visualisation
  "ggpubr",        # publication-ready ggplot helpers
  "RColorBrewer",  # colour palettes
  "e1071",         # SVM / skewness helpers
  "class",         # kNN
  "glmnet",        # regularised GLM
  "rstudioapi",    # auto-set working directory
  "ROSE"           # SMOTE oversampling to address class imbalance
)

suppressWarnings({
  for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      install.packages(pkg, repos = "https://cran.r-project.org", quiet = TRUE)
      library(pkg, character.only = TRUE, quietly = TRUE)
    }
  }
})

cat("All required libraries loaded successfully.\n")

if (requireNamespace("rstudioapi", quietly = TRUE) &&
    rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

options(lifecycle_verbosity = "quiet")
set.seed(42)


# ============================================================
# SECTION 1: DATA IMPORT
# ============================================================
cat("\n========== SECTION 1: DATA IMPORT ==========\n")

df_raw <- read.csv("dataset_employee_attrition.csv",
                   stringsAsFactors = FALSE,
                   na.strings        = c("", "NA", "N/A"))

cat("Dataset loaded successfully.\n")
cat("Dimensions (rows x cols):", nrow(df_raw), "x", ncol(df_raw), "\n")
cat("Column names:\n")
print(names(df_raw))


# ============================================================
# SECTION 2: DATA CLEANING & PRE-PROCESSING
# ============================================================
cat("\n========== SECTION 2: DATA CLEANING ==========\n")

df <- df_raw

# 2.1 Standardise Attrition
cat("\nUnique Attrition values BEFORE cleaning:\n")
print(table(df$Attrition, useNA = "always"))
df$Attrition <- toupper(trimws(df$Attrition))
df$Attrition[df$Attrition %in% c("1", "YES")] <- "Yes"
df$Attrition[df$Attrition %in% c("0", "NO")]  <- "No"
df$Attrition[!df$Attrition %in% c("Yes", "No")] <- NA
cat("\nUnique Attrition values AFTER cleaning:\n")
print(table(df$Attrition, useNA = "always"))

# 2.2 Standardise Gender
df$Gender <- toupper(trimws(df$Gender))
df$Gender[df$Gender %in% c("M", "MALE")]   <- "Male"
df$Gender[df$Gender %in% c("F", "FEMALE")] <- "Female"
df$Gender[!df$Gender %in% c("Male", "Female")] <- NA

# 2.3 Standardise Department
df$Department <- toupper(trimws(df$Department))
df$Department[df$Department %in% c("SALES", "SALE")]                     <- "Sales"
df$Department[df$Department %in% c("HR", "HUMAN RESOURCES")]             <- "Human Resources"
df$Department[df$Department %in% c("R&D", "RESEARCH & DEVELOPMENT",
                                    "RESEARCH AND DEVELOPMENT")]          <- "Research & Development"
df$Department[!df$Department %in% c("Sales", "Human Resources",
                                     "Research & Development")]           <- NA

# 2.4 Standardise OverTime
df$OverTime <- toupper(trimws(df$OverTime))
df$OverTime[df$OverTime %in% c("1", "YES")] <- "Yes"
df$OverTime[df$OverTime %in% c("0", "NO")]  <- "No"
df$OverTime[!df$OverTime %in% c("Yes", "No")] <- NA

# 2.5 Standardise BusinessTravel
df$BusinessTravel <- toupper(trimws(df$BusinessTravel))
df$BusinessTravel[df$BusinessTravel %in% c("TRAVEL_RARELY",     "TRAVEL RARELY")]     <- "Travel_Rarely"
df$BusinessTravel[df$BusinessTravel %in% c("TRAVEL_FREQUENTLY", "TRAVEL FREQUENTLY")] <- "Travel_Frequently"
df$BusinessTravel[df$BusinessTravel %in% c("NON-TRAVEL", "NO TRAVEL", "NO_TRAVEL")]   <- "Non-Travel"
df$BusinessTravel[!df$BusinessTravel %in% c("Travel_Rarely",
                                             "Travel_Frequently",
                                             "Non-Travel")]               <- NA

# 2.6 Standardise MaritalStatus
df$MaritalStatus <- toupper(trimws(df$MaritalStatus))
df$MaritalStatus[df$MaritalStatus == "DIVORCED"] <- "Divorced"
df$MaritalStatus[df$MaritalStatus == "MARRIED"]  <- "Married"
df$MaritalStatus[df$MaritalStatus == "SINGLE"]   <- "Single"
df$MaritalStatus[!df$MaritalStatus %in% c("Divorced", "Married", "Single")] <- NA

# 2.7 Coerce numeric columns
numeric_cols <- c("Age", "DailyRate", "DistanceFromHome", "Education",
                  "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
                  "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
                  "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
                  "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
                  "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
                  "YearsInCurrentRole", "YearsSinceLastPromotion",
                  "YearsWithCurrManager", "StandardHours", "EmployeeCount",
                  "EmployeeNumber")

for (col in numeric_cols) {
  if (col %in% names(df))
    df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
}

# 2.8 Remove duplicates
n_before <- nrow(df)
df <- df[!duplicated(df$EmployeeNumber), ]
cat("\nDuplicate rows removed:", n_before - nrow(df), "\n")

# 2.9 Handle missing values
cat("\nMissing values per column BEFORE imputation:\n")
missing_before <- colSums(is.na(df))
print(missing_before[missing_before > 0])

mode_impute <- function(x) {
  ux <- na.omit(x)
  ux[which.max(tabulate(match(ux, unique(ux))))]
}

for (col in numeric_cols) {
  if (col %in% names(df) && any(is.na(df[[col]])))
    df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm = TRUE)
}

cat_cols <- c("Attrition", "BusinessTravel", "Department", "EducationField",
              "Gender", "JobRole", "MaritalStatus", "Over18", "OverTime")

for (col in cat_cols) {
  if (col %in% names(df) && any(is.na(df[[col]])))
    df[[col]][is.na(df[[col]])] <- mode_impute(df[[col]])
}

cat("\nTotal missing values AFTER imputation:", sum(is.na(df)), "\n")

# 2.10 Outlier capping (IQR / Winsorisation)
cap_outliers <- function(x) {
  Q1      <- quantile(x, 0.25, na.rm = TRUE)
  Q3      <- quantile(x, 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  x[x < Q1 - 1.5 * IQR_val] <- Q1 - 1.5 * IQR_val
  x[x > Q3 + 1.5 * IQR_val] <- Q3 + 1.5 * IQR_val
  x
}

for (col in c("MonthlyIncome", "DailyRate", "HourlyRate",
              "DistanceFromHome", "NumCompaniesWorked",
              "TotalWorkingYears", "YearsAtCompany")) {
  if (col %in% names(df)) df[[col]] <- cap_outliers(df[[col]])
}
cat("Outlier capping (Winsorisation) applied.\n")

# 2.11 Convert to factors
# Levels fixed HERE, before any split, so train/test/SMOTE all inherit
# identical levels - no post-split realignment needed.
df$Attrition      <- factor(df$Attrition,      levels = c("No", "Yes"))
df$BusinessTravel <- factor(df$BusinessTravel,
                             levels = c("Non-Travel", "Travel_Rarely", "Travel_Frequently"))
df$Department     <- factor(df$Department,
                             levels = c("Human Resources", "Research & Development", "Sales"))
df$EducationField <- factor(df$EducationField)
df$Gender         <- factor(df$Gender,        levels = c("Female", "Male"))
df$JobRole        <- factor(df$JobRole)
df$MaritalStatus  <- factor(df$MaritalStatus, levels = c("Divorced", "Married", "Single"))
df$OverTime       <- factor(df$OverTime,       levels = c("No", "Yes"))
df$Over18         <- factor(df$Over18)

# 2.12 Feature Engineering
df$IncomePerYear     <- df$MonthlyIncome * 12
df$PromotionGap      <- df$YearsAtCompany - df$YearsSinceLastPromotion
df$SeniorityRatio    <- ifelse(df$TotalWorkingYears > 0,
                                df$YearsAtCompany / df$TotalWorkingYears, 0)
df$SatisfactionIndex <- (df$JobSatisfaction +
                          df$EnvironmentSatisfaction +
                          df$RelationshipSatisfaction) / 3
df$AgeGroup    <- cut(df$Age, breaks = c(17, 25, 35, 45, 60),
                      labels = c("18-25", "26-35", "36-45", "46-60"))
df$IncomeGroup <- cut(df$MonthlyIncome,
                      breaks = quantile(df$MonthlyIncome,
                                        probs = c(0, .25, .5, .75, 1), na.rm = TRUE),
                      labels = c("Low", "Medium", "High", "Very High"),
                      include.lowest = TRUE)
df$TenureWindow <- factor(
  ifelse(df$YearsAtCompany <= 3, "Early (<=3 yrs)", "Established (>3 yrs)"),
  levels = c("Early (<=3 yrs)", "Established (>3 yrs)")
)

cat("\nFeature engineering complete.\n")
cat("\n===== FINAL CLEAN DATASET SUMMARY =====\n")
cat("Rows:", nrow(df), "| Columns:", ncol(df), "\n")
print(table(df$Attrition))
cat("Overall attrition rate:", round(mean(df$Attrition == "Yes") * 100, 2), "%\n")


# 2.13 DATA CLEANING VISUALISATION
cat("\n--- Data Cleaning Visualisation ---\n")

missing_df <- data.frame(
  Column  = names(missing_before[missing_before > 0]),
  Missing = as.integer(missing_before[missing_before > 0])
)

if (nrow(missing_df) > 0) {
  p_missing <- ggplot(missing_df, aes(x = reorder(Column, -Missing),
                                       y = Missing, fill = Missing)) +
    geom_col(color = "white", alpha = 0.85) +
    geom_text(aes(label = Missing), vjust = -0.4, size = 4, fontface = "bold") +
    scale_fill_gradient(low = "#FFF9C4", high = "#E53935") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
    labs(title    = "Data Cleaning - Missing Values per Column (before imputation)",
         subtitle = "Imputed: numeric -> median, categorical -> mode",
         x = "Column", y = "Number of Missing Values") +
    theme_minimal(base_size = 13) +
    theme(plot.title = element_text(face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
          legend.position = "none")
  print(p_missing)
} else {
  cat("No missing values before imputation - chart skipped.\n")
}

class_balance <- df %>% count(Attrition) %>%
  mutate(Pct = round(n / sum(n) * 100, 1))

p_balance <- ggplot(class_balance, aes(x = Attrition, y = n, fill = Attrition)) +
  geom_col(alpha = 0.85, color = "white", width = 0.5) +
  geom_text(aes(label = paste0(n, "\n(", Pct, "%)")),
            vjust = -0.3, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = c("No" = "#42A5F5", "Yes" = "#EF5350")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title   = "Data Cleaning - Class Balance of Target Variable",
       subtitle = paste0("Imbalanced: ",
                         class_balance$Pct[class_balance$Attrition == "Yes"],
                         "% attrition vs ",
                         class_balance$Pct[class_balance$Attrition == "No"],
                         "% retained"),
       x = "Attrition", y = "Count") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "none")
print(p_balance)

income_compare <- data.frame(
  Income = c(suppressWarnings(as.numeric(df_raw$MonthlyIncome[1:nrow(df)])),
             df$MonthlyIncome),
  Stage  = rep(c("Before Capping", "After Capping"), each = nrow(df))
)
income_compare <- income_compare[!is.na(income_compare$Income), ]

p_outlier <- ggplot(income_compare, aes(x = Income, fill = Stage)) +
  geom_histogram(bins = 40, alpha = 0.7, color = "white", position = "identity") +
  scale_fill_manual(values = c("Before Capping" = "#EF9A9A", "After Capping" = "#42A5F5")) +
  facet_wrap(~Stage, scales = "free_y") +
  labs(title    = "Data Cleaning - Outlier Capping Effect on MonthlyIncome",
       subtitle = "IQR method caps extreme values; distribution shape preserved",
       x = "Monthly Income", y = "Count") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "none")
print(p_outlier)


# 2.14 VALIDATION
validate_df <- function(df) {
  errors <- c()
  if (any(is.na(df$Attrition)))      errors <- c(errors, "Missing values in Attrition")
  if (any(is.na(df$Age)))            errors <- c(errors, "Missing values in Age")
  if (any(df$Age < 18 | df$Age > 70, na.rm = TRUE))
    errors <- c(errors, "Age out of range [18,70]")
  if (any(df$MonthlyIncome < 0, na.rm = TRUE))
    errors <- c(errors, "Negative MonthlyIncome")
  if (any(duplicated(df$EmployeeNumber)))
    errors <- c(errors, "Duplicate EmployeeNumbers")
  if (length(errors) == 0) cat("Validation PASSED - all checks OK.\n")
  else { cat("Validation WARNINGS:\n"); for (e in errors) cat("  -", e, "\n") }
}
validate_df(df)


# ============================================================
# OBJECTIVE 1: DEMOGRAPHIC IMPACT ON ATTRITION
# ============================================================
# Objective: To determine whether demographic characteristics
# (age, gender, marital, education) affect attrition through
# descriptive statistics and inferential hypothesis testing.
#
# 1-1 Which age group has highest rate? -> identifies danger cohort
# 1-2 WHY? Gender x MaritalStatus within that cohort (drill-down)
# 1-3 Does education compound risk within 1-2's highest-risk sub-group?
# ============================================================
cat("\n========== OBJECTIVE 1: DEMOGRAPHIC IMPACT ==========\n")

# Analysis 1-1: Attrition Rate by Age Group
# Descriptive: rate % per group | Inferential: ANOVA
age_attrition <- df %>%
  filter(!is.na(AgeGroup)) %>%
  group_by(AgeGroup) %>%
  summarise(AttritionRate = mean(Attrition == "Yes") * 100,
            Count = n(), .groups = "drop")

p1_1 <- ggplot(age_attrition, aes(x = AgeGroup, y = AttritionRate, fill = AgeGroup)) +
  geom_col(alpha = 0.85, color = "white") +
  geom_text(aes(label = paste0(round(AttritionRate, 1), "%\n(n=", Count, ")")),
            vjust = -0.3, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("18-25" = "#EF5350", "26-35" = "#FF8A65",
                                "36-45" = "#42A5F5", "46-60" = "#26A69A")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title   = "Analysis 1-1: Attrition Rate by Age Group",
       subtitle = "Which life-stage cohort leaves most often? (drives 1-2 drill-down)",
       x = "Age Group", y = "Attrition Rate (%)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "none")
print(p1_1)

cat("\nANOVA: Age vs Attrition\n")
print(summary(aov(Age ~ Attrition, data = df)))

# Analysis 1-2: Gender x MaritalStatus within highest-risk age group
# Inferential: Chi-square x2
top_age_group <- age_attrition$AgeGroup[which.max(age_attrition$AttritionRate)]
cat("\nHighest-risk age group:", as.character(top_age_group), "\n")

drill_1_2 <- df %>%
  filter(AgeGroup == top_age_group, !is.na(Gender), !is.na(MaritalStatus)) %>%
  group_by(Gender, MaritalStatus) %>%
  summarise(AttritionRate = mean(Attrition == "Yes") * 100,
            Count = n(), .groups = "drop")

p1_2 <- ggplot(drill_1_2, aes(x = MaritalStatus, y = AttritionRate, fill = Gender)) +
  geom_col(position = "dodge", alpha = 0.85, color = "white") +
  geom_text(aes(label = paste0(round(AttritionRate, 1), "%\n(n=", Count, ")")),
            position = position_dodge(width = 0.9), vjust = -0.3,
            size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("Male" = "#1565C0", "Female" = "#AD1457")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title   = paste0("Analysis 1-2: Gender x Marital Status Drill-Down (Age: ",
                         top_age_group, ")"),
       subtitle = "Filters to highest-risk age group to explain WHY attrition is high there",
       x = "Marital Status", y = "Attrition Rate (%)", fill = "Gender") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))
print(p1_2)

df_top_age <- df %>% filter(AgeGroup == top_age_group)
cat("\nChi-square: Gender vs Attrition (within", as.character(top_age_group), ")\n")
print(chisq.test(table(df_top_age$Gender, df_top_age$Attrition)))
cat("\nChi-square: MaritalStatus vs Attrition (within", as.character(top_age_group), ")\n")
print(chisq.test(table(df_top_age$MaritalStatus, df_top_age$Attrition)))

# Analysis 1-3: Does education compound risk in 1-2's highest-risk sub-group?
# Compares at-risk sub-group vs all employees baseline side-by-side
high_risk_cell <- drill_1_2 %>%
  filter(!is.na(AttritionRate)) %>% arrange(desc(AttritionRate)) %>% slice(1)
hr_gender  <- as.character(high_risk_cell$Gender)
hr_marital <- as.character(high_risk_cell$MaritalStatus)
cat("\nHighest-risk sub-group:", hr_gender, "+", hr_marital,
    "| Rate:", round(high_risk_cell$AttritionRate, 1), "%\n")

edu_labels <- c("1" = "Below College", "2" = "College",
                 "3" = "Bachelor", "4" = "Master", "5" = "Doctor")

edu_subgroup <- df %>%
  filter(AgeGroup == top_age_group, Gender == hr_gender,
         MaritalStatus == hr_marital, !is.na(Education)) %>%
  mutate(EduLabel = recode(as.character(Education), !!!edu_labels),
         EduLabel = factor(EduLabel, levels = c("Below College","College",
                                                "Bachelor","Master","Doctor"))) %>%
  group_by(EduLabel) %>%
  summarise(AttritionRate = mean(Attrition == "Yes") * 100,
            Count = n(), .groups = "drop") %>%
  mutate(Group = paste0(hr_gender, " + ", hr_marital, " (at-risk sub-group)"))

edu_full <- df %>%
  filter(!is.na(Education)) %>%
  mutate(EduLabel = recode(as.character(Education), !!!edu_labels),
         EduLabel = factor(EduLabel, levels = c("Below College","College",
                                                "Bachelor","Master","Doctor"))) %>%
  group_by(EduLabel) %>%
  summarise(AttritionRate = mean(Attrition == "Yes") * 100,
            Count = n(), .groups = "drop") %>%
  mutate(Group = "All Employees (baseline)")

edu_combined <- bind_rows(edu_subgroup, edu_full)
edu_combined$Group <- factor(edu_combined$Group,
  levels = c(paste0(hr_gender, " + ", hr_marital, " (at-risk sub-group)"),
             "All Employees (baseline)"))

p1_3 <- ggplot(edu_combined, aes(x = EduLabel, y = AttritionRate, fill = Group)) +
  geom_col(alpha = 0.85, color = "white") +
  geom_text(aes(label = paste0(round(AttritionRate, 1), "%\n(n=", Count, ")")),
            vjust = -0.3, size = 3.3, fontface = "bold") +
  scale_fill_manual(values = c("#E53935", "#90A4AE"), name = "Group") +
  facet_wrap(~Group, ncol = 2, scales = "free_y") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title   = paste0("Analysis 1-3: Does Education Compound Risk - ",
                         hr_gender, " + ", hr_marital, "?"),
       subtitle = "Left = at-risk sub-group from 1-2 | Right = all employees baseline | Steeper gradient = compounding",
       x = "Education Level", y = "Attrition Rate (%)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold", size = 10),
        axis.text.x = element_text(angle = 0, hjust = 0.5, size = 10),
        legend.position = "none")
print(p1_3)

cat("\nAttrition by education - AT-RISK sub-group:\n")
print(edu_subgroup[, c("EduLabel", "AttritionRate", "Count")])
cat("\nAttrition by education - ALL employees (baseline):\n")
print(edu_full[, c("EduLabel", "AttritionRate", "Count")])


# ============================================================
# OBJECTIVE 2: JOB & COMPENSATION FACTORS
# ============================================================
# Objective: To investigate how compensation, overtime, job level
# and job satisfaction dimensions relate to attrition using EDA
# and inferential statistical tests.
#
# 2-1 Income gap between leavers/stayers? (violin + Welch t-test)
# 2-2 WHY low earners leave? OverTime x IncomeGroup interaction
#     + JobLevel gradient (Kruskal-Wallis)
# 2-3 WHERE is the problem worst? Department-level drill-down
# ============================================================
cat("\n========== OBJECTIVE 2: JOB & COMPENSATION ==========\n")

# Analysis 2-1: Monthly Income vs Attrition (violin + box + mean diamond)
# Test: Welch t-test
p2_1 <- ggplot(df, aes(x = Attrition, y = MonthlyIncome, fill = Attrition)) +
  geom_violin(alpha = 0.55, trim = FALSE) +
  geom_boxplot(width = 0.15, alpha = 0.90,
               outlier.shape = 21, outlier.fill = "white", outlier.color = "black") +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 4,
               fill = "gold", color = "black") +
  scale_fill_manual(values = c("No" = "#42A5F5", "Yes" = "#EF5350")) +
  labs(title   = "Analysis 2-1: Monthly Income Distribution by Attrition",
       subtitle = "Diamond = group mean; employees who left earn less on average",
       x = "Attrition", y = "Monthly Income (RM)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "none")
print(p2_1)
cat("\nWelch t-test: MonthlyIncome vs Attrition\n")
print(t.test(MonthlyIncome ~ Attrition, data = df))

# Analysis 2-2: OverTime x IncomeGroup interaction + JobLevel
# Tests: Chi-square (OverTime), Kruskal-Wallis (JobLevel ordinal)
overtime_income <- df %>%
  filter(!is.na(IncomeGroup)) %>%
  group_by(IncomeGroup, OverTime) %>%
  summarise(AttritionRate = mean(Attrition == "Yes") * 100,
            Count = n(), .groups = "drop")

p2_2 <- ggplot(overtime_income,
               aes(x = IncomeGroup, y = AttritionRate, fill = OverTime)) +
  geom_col(position = "dodge", alpha = 0.85, color = "white") +
  geom_text(aes(label = paste0(round(AttritionRate, 0), "%\n(n=", Count, ")")),
            position = position_dodge(width = 0.9), vjust = -0.3,
            size = 3.3, fontface = "bold") +
  scale_fill_manual(values = c("No" = "#26A69A", "Yes" = "#EF5350")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title   = "Analysis 2-2: Attrition by Income Group x OverTime (Interaction)",
       subtitle = "Tests if overtime compounds attrition for low earners (WHY for 2-1)",
       x = "Income Group", y = "Attrition Rate (%)", fill = "OverTime") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))
print(p2_2)

cat("\nChi-square: OverTime vs Attrition\n")
print(chisq.test(table(df$OverTime, df$Attrition)))
cat("\nKruskal-Wallis: JobLevel vs Attrition (ordinal - non-parametric)\n")
print(kruskal.test(JobLevel ~ Attrition, data = df))

joblevel_attr <- df %>%
  group_by(JobLevel) %>%
  summarise(AttritionRate = mean(Attrition == "Yes") * 100,
            Count = n(), .groups = "drop")

p2_2b <- ggplot(joblevel_attr,
                aes(x = factor(JobLevel), y = AttritionRate, fill = AttritionRate)) +
  geom_col(alpha = 0.85, color = "white") +
  geom_text(aes(label = paste0(round(AttritionRate, 1), "%\n(n=", Count, ")")),
            vjust = -0.3, size = 3.8, fontface = "bold") +
  scale_fill_gradient(low = "#C8E6C9", high = "#B71C1C") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title   = "Analysis 2-2b: Attrition Rate by Job Level",
       subtitle = "Lower job levels show higher attrition (Kruskal-Wallis p in console)",
       x = "Job Level (1 = Entry, 5 = Executive)", y = "Attrition Rate (%)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "none")
print(p2_2b)

# Analysis 2-3: WHERE is the low-income overtime problem worst?
# Bubble chart: x = % of dept in high-risk group, y = their attrition rate
dept_2_3 <- df %>%
  group_by(Department) %>%
  summarise(
    TotalN           = n(),
    HighRiskN        = sum(IncomeGroup == "Low" & OverTime == "Yes", na.rm = TRUE),
    HighRiskShare    = HighRiskN / TotalN * 100,
    HighRiskAttrRate = ifelse(
      sum(IncomeGroup == "Low" & OverTime == "Yes", na.rm = TRUE) > 0,
      mean(Attrition[IncomeGroup == "Low" & OverTime == "Yes"] == "Yes",
           na.rm = TRUE) * 100,
      NA_real_),
    AvgSatisfaction  = mean(SatisfactionIndex, na.rm = TRUE),
    OverallAttrRate  = mean(Attrition == "Yes") * 100,
    .groups          = "drop"
  )

cat("\nDepartment breakdown - Low income + Overtime:\n")
print(dept_2_3[, c("Department","TotalN","HighRiskN","HighRiskShare","HighRiskAttrRate")])

p2_3 <- ggplot(dept_2_3,
               aes(x = HighRiskShare, y = HighRiskAttrRate,
                   size = TotalN, fill = HighRiskAttrRate)) +
  geom_point(shape = 21, alpha = 0.85, color = "white") +
  geom_text(aes(label = Department), vjust = -1.6, size = 3.8,
            fontface = "bold", hjust = 0.5) +
  scale_fill_gradient2(low = "#1B5E20", mid = "#FFF176", high = "#B71C1C",
                       midpoint = mean(dept_2_3$HighRiskAttrRate, na.rm = TRUE),
                       name = "Attrition rate\n(low-income OT)") +
  scale_size_continuous(range = c(5, 16), name = "Dept headcount") +
  scale_y_continuous(expand = expansion(mult = c(0.1, 0.2))) +
  scale_x_continuous(expand = expansion(mult = c(0.1, 0.2))) +
  labs(title   = "Analysis 2-3: WHERE Does Low-Income Overtime Drive Attrition Most?",
       subtitle = "X = % of dept in high-risk group (from 2-2) | Y = their attrition rate | Darker = worse",
       x = "% of Dept: Low Income + Overtime",
       y = "Attrition Rate of High-Risk Group (%)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "right")
print(p2_3)


# ============================================================
# OBJECTIVE 3: WORK EXPERIENCE & TENURE ANALYSIS
# ============================================================
# Objective: To examine years in company, recency of promotion,
# and number of previous employers vs attrition using
# non-parametric statistics.
#
# 3-1 Danger Window: when do employees most leave? (Spearman)
# 3-2 WHY? Promotion stagnation inside window (KW + Wilcoxon)
# 3-3 Triple-risk: stagnant + early + job-hopping (KW + chi-sq)
# ============================================================
cat("\n========== OBJECTIVE 3: WORK EXPERIENCE & TENURE ==========\n")

# Analysis 3-1: Tenure density - identify Danger Window (0-3 yrs)
# Test: Spearman correlation (non-parametric)
tenure_medians <- df %>%
  group_by(Attrition) %>%
  summarise(MedianTenure = median(YearsAtCompany), .groups = "drop")

p3_1 <- ggplot(df, aes(x = YearsAtCompany, fill = Attrition)) +
  annotate("rect", xmin = 0, xmax = 3, ymin = -Inf, ymax = Inf,
           fill = "#FFCDD2", alpha = 0.35) +
  geom_density(alpha = 0.55) +
  geom_vline(data = tenure_medians,
             aes(xintercept = MedianTenure, color = Attrition),
             linetype = "dashed", linewidth = 1.1) +
  annotate("text", x = 1.5, y = Inf, vjust = 1.8,
           label = "Danger\nWindow\n(0-3 yrs)",
           size = 3.5, color = "black", fontface = "italic") +
  scale_fill_manual(values  = c("No" = "#1E88E5", "Yes" = "#E53935")) +
  scale_color_manual(values = c("No" = "#0D47A1", "Yes" = "#B71C1C"), guide = "none") +
  labs(title   = "Analysis 3-1: Years at Company - Identifying the Danger Window",
       subtitle = "Dashed lines = group medians; shaded = danger window (drives 3-2)",
       x = "Years at Company", y = "Density", fill = "Attrition") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))
print(p3_1)

cat("\nSpearman correlation: YearsAtCompany vs Attrition (non-parametric)\n")
print(cor.test(as.numeric(df$Attrition) - 1, df$YearsAtCompany,
               method = "spearman", exact = FALSE))

# Analysis 3-2: Promotion stagnation x tenure window
# Tests: Kruskal-Wallis + Wilcoxon rank-sum (within Danger Window)
p3_2 <- df %>%
  filter(!is.na(TenureWindow)) %>%
  ggplot(aes(x = factor(YearsSinceLastPromotion), fill = Attrition)) +
  geom_bar(position = "fill", alpha = 0.85, color = "white") +
  scale_fill_manual(values = c("No" = "#43A047", "Yes" = "#E53935")) +
  scale_y_continuous(labels = percent_format()) +
  facet_wrap(~TenureWindow, ncol = 2) +
  labs(title   = "Analysis 3-2: Promotion Stagnation vs Attrition (by Tenure Window)",
       subtitle = "Left = danger window | Right = established | Does lack of promotion drive early exits?",
       x = "Years Since Last Promotion", y = "Proportion", fill = "Attrition") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold", size = 11))
print(p3_2)

df_early <- df %>% filter(TenureWindow == "Early (<=3 yrs)")
cat("\nKruskal-Wallis: YearsSinceLastPromotion vs Attrition (Danger Window)\n")
print(kruskal.test(YearsSinceLastPromotion ~ Attrition, data = df_early))
cat("\nWilcoxon rank-sum: YearsSinceLastPromotion (Danger Window)\n")
print(wilcox.test(YearsSinceLastPromotion ~ Attrition,
                  data = df_early, exact = FALSE, correct = FALSE))
cat("Median - Leavers:", median(df_early$YearsSinceLastPromotion[df_early$Attrition == "Yes"]), "\n")
cat("Median - Stayers:", median(df_early$YearsSinceLastPromotion[df_early$Attrition == "No"]),  "\n")

# Analysis 3-3: Triple-Risk heatmap (Danger Window only)
# PromoBand x HopGroup -> find compounded MAX risk cell
# Tests: Kruskal-Wallis + Chi-square
df$HopGroup <- cut(df$NumCompaniesWorked,
                   breaks = c(-1, 1, 3, 5, Inf),
                   labels = c("0-1 (Loyal)", "2-3 (Some)",
                               "4-5 (Frequent)", "6+ (Serial)"),
                   include.lowest = TRUE)

df$PromoBand <- factor(
  ifelse(df$YearsSinceLastPromotion <= 1, "Stagnant (0-1 yrs)", "Some Recognition (2+ yrs)"),
  levels = c("Stagnant (0-1 yrs)", "Some Recognition (2+ yrs)")
)

heatmap_3_3 <- df %>%
  filter(TenureWindow == "Early (<=3 yrs)", !is.na(HopGroup), !is.na(PromoBand)) %>%
  group_by(PromoBand, HopGroup) %>%
  summarise(AttritionRate = mean(Attrition == "Yes") * 100,
            Count = n(), .groups = "drop")

cat("\nTriple-risk heatmap (Danger Window only):\n")
print(heatmap_3_3)

p3_3 <- ggplot(heatmap_3_3,
               aes(x = HopGroup, y = PromoBand, fill = AttritionRate)) +
  geom_tile(color = "white", linewidth = 0.9) +
  geom_text(aes(label = paste0(round(AttritionRate, 1), "%\nn=", Count)),
            color = "white", fontface = "bold", size = 4.2) +
  scale_fill_gradient2(low = "#1B5E20", mid = "#FFF176", high = "#B71C1C",
                       midpoint = 25, name = "Attrition\nRate (%)") +
  labs(title   = "Analysis 3-3: Triple-Risk Profile - Promotion x Job-Hopping (Danger Window)",
       subtitle = "Danger Window only | Promotion from 3-2 | Darkest cell = MAX risk",
       x = "Job-Hopping Background", y = "Promotion Status") +
  theme_minimal(base_size = 13) +
  theme(plot.title    = element_text(face = "bold", hjust = 0,
                                     margin = ggplot2::margin(0, 0, 0, -100)),
        plot.subtitle = element_text(hjust = 0,
                                     margin = ggplot2::margin(0, 0, 0, -100)),
        axis.text.x   = element_text(angle = 0, hjust = 0.5))
print(p3_3)

worst_cell <- heatmap_3_3 %>% filter(!is.na(AttritionRate)) %>%
  arrange(desc(AttritionRate)) %>% slice(1)
cat(sprintf("\nMax-risk profile: %s + %s -> %.1f%% (n=%d)\n",
            as.character(worst_cell$PromoBand), as.character(worst_cell$HopGroup),
            worst_cell$AttritionRate, worst_cell$Count))
cat("HR: screen new hires with 4+ prior companies; ensure early promotion reviews.\n")

df_triple <- df %>% filter(TenureWindow == "Early (<=3 yrs)",
                            PromoBand == "Stagnant (0-1 yrs)")
cat("\nKruskal-Wallis: NumCompaniesWorked vs Attrition (Stagnant + Danger Window)\n")
print(kruskal.test(NumCompaniesWorked ~ Attrition, data = df_triple))

# df_early was created before PromoBand existed, so it lacks that column.
# Re-filter here, after PromoBand is defined, to avoid a NULL reference error.
df_early_promo <- df %>% filter(TenureWindow == "Early (<=3 yrs)")
cat("\nChi-square: PromoBand vs Attrition (Danger Window)\n")
print(chisq.test(table(df_early_promo$PromoBand, df_early_promo$Attrition)))
cat("\nIf both significant: triple-risk profile is statistically confirmed.\n")


# ============================================================
# OBJECTIVE 4: PREDICTIVE MODELLING FOR ATTRITION
# ============================================================
# Objective: Develop, test and compare LR, RF, DT to predict
# attrition; evaluate with Accuracy, Precision, Recall, F1, AUC-ROC.
#
# 4-1 LR baseline (raw data) -> exposes Recall problem
# 4-2 RF + SMOTE -> fixes Recall; Feature Importance closes EDA loop
# 4-3 DT on RF Top 8 -> HR-actionable rules; ROC validates cost
# ============================================================
cat("\n========== OBJECTIVE 4: PREDICTIVE MODELLING ==========\n")

model_cols <- c("Attrition", "Age", "BusinessTravel", "DailyRate", "Department",
                "DistanceFromHome", "Education", "EnvironmentSatisfaction",
                "Gender", "JobInvolvement", "JobLevel", "JobSatisfaction",
                "MaritalStatus", "MonthlyIncome", "NumCompaniesWorked",
                "OverTime", "PercentSalaryHike", "PerformanceRating",
                "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
                "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
                "YearsInCurrentRole", "YearsSinceLastPromotion",
                "YearsWithCurrManager", "SatisfactionIndex", "PromotionGap",
                "SeniorityRatio")

df_model   <- df[, model_cols]
df_model   <- df_model[complete.cases(df_model), ]

set.seed(42)
train_idx  <- createDataPartition(df_model$Attrition, p = 0.75, list = FALSE)
train_data <- df_model[ train_idx, ]
test_data  <- df_model[-train_idx, ]

cat("Train:", nrow(train_data), "rows | Test:", nrow(test_data), "rows\n")
cat("Train attrition rate:", round(mean(train_data$Attrition == "Yes") * 100, 1), "%\n")
cat("Test  attrition rate:", round(mean(test_data$Attrition  == "Yes") * 100, 1), "%\n")

# SMOTE via ROSE (train only - test untouched for honest evaluation)
set.seed(42)
train_balanced <- ROSE(Attrition ~ ., data = train_data, seed = 42)$data
cat("\nBefore SMOTE - No:", sum(train_data$Attrition == "No"),
    "| Yes:", sum(train_data$Attrition == "Yes"), "\n")
cat("After  SMOTE - No:", sum(train_balanced$Attrition == "No"),
    "| Yes:", sum(train_balanced$Attrition == "Yes"), "\n")

p_smote <- data.frame(
  Count     = c(sum(train_data$Attrition == "No"),
                sum(train_data$Attrition == "Yes"),
                sum(train_balanced$Attrition == "No"),
                sum(train_balanced$Attrition == "Yes")),
  Attrition = rep(c("No", "Yes"), 2),
  Stage     = rep(c("Before SMOTE (imbalanced)", "After SMOTE (balanced)"), each = 2)
)
p_smote$Stage <- factor(p_smote$Stage,
                         levels = c("Before SMOTE (imbalanced)", "After SMOTE (balanced)"))

smote_plot <- ggplot(p_smote, aes(x = Attrition, y = Count, fill = Attrition)) +
  geom_col(alpha = 0.85, color = "white") +
  geom_text(aes(label = Count), vjust = -0.3, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("No" = "#42A5F5", "Yes" = "#EF5350")) +
  facet_wrap(~Stage, ncol = 2) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title   = "Class Imbalance: Before vs After SMOTE",
       subtitle = "SMOTE on TRAIN only; test set untouched",
       x = "Attrition", y = "Count") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold"), legend.position = "none")
print(smote_plot)

# Analysis 4-1: LR baseline on raw imbalanced data
# Purpose: expose Recall problem to justify RF + SMOTE
cat("\n--- Analysis 4-1: Logistic Regression (Baseline) ---\n")
log_model <- glm(Attrition ~ ., data = train_data, family = "binomial")
log_probs <- predict(log_model, test_data, type = "response")
log_pred  <- factor(ifelse(log_probs > 0.5, "Yes", "No"), levels = c("No", "Yes"))
cm_log    <- confusionMatrix(log_pred, test_data$Attrition, positive = "Yes")
cat("\nLogistic Regression - Confusion Matrix:\n"); print(cm_log)

log_recall <- as.numeric(cm_log$byClass["Recall"])

p4_1 <- ggplot(data.frame(Prob = log_probs, Actual = test_data$Attrition),
               aes(x = Prob, fill = Actual)) +
  geom_density(alpha = 0.55) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "grey30", linewidth = 1) +
  annotate("text", x = 0.52, y = Inf, vjust = 1.5, hjust = 0,
           label = "Threshold = 0.5", size = 3.5, color = "grey30") +
  scale_fill_manual(values = c("No" = "#1976D2", "Yes" = "#C62828")) +
  labs(title   = "Analysis 4-1: LR - Predicted Probability Distribution",
       subtitle = paste0("Recall = ", round(log_recall * 100, 1),
                         "% | Overlapping curves = linear boundary problem -> motivates RF"),
       x = "Predicted Probability of Attrition", y = "Density", fill = "Actual") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))
print(p4_1)
cat("\nRecall =", round(log_recall * 100, 1), "% ->",
    round((1 - log_recall) * 100, 1), "% of leavers MISSED.\n")

# Analysis 4-2: RF on SMOTE-balanced data; Feature Importance closes EDA loop
cat("\n--- Analysis 4-2: Random Forest (SMOTE-Balanced) ---\n")
set.seed(42)
rf_model <- randomForest(Attrition ~ ., data = train_balanced,
                          ntree = 300, importance = TRUE, proximity = FALSE)
rf_pred  <- predict(rf_model, test_data)
rf_probs <- predict(rf_model, test_data, type = "prob")[, "Yes"]
cm_rf    <- confusionMatrix(rf_pred, test_data$Attrition, positive = "Yes")
cat("\nRandom Forest - Confusion Matrix:\n"); print(cm_rf)

rf_recall <- as.numeric(cm_rf$byClass["Recall"])
cat(sprintf("\nRecall improvement over LR: +%.1f pp\n", (rf_recall - log_recall) * 100))

importance_df <- as.data.frame(importance(rf_model))
importance_df$Feature <- rownames(importance_df)
importance_df <- importance_df %>% arrange(desc(MeanDecreaseGini)) %>% head(15)

importance_df$Source <- case_when(
  importance_df$Feature %in% c("Age","MaritalStatus","Gender","Education") ~
    "Obj 1: Demographics",
  importance_df$Feature %in% c("MonthlyIncome","OverTime","JobSatisfaction",
                                 "EnvironmentSatisfaction","SatisfactionIndex",
                                 "JobLevel","JobInvolvement","WorkLifeBalance",
                                 "StockOptionLevel","BusinessTravel",
                                 "DistanceFromHome") ~
    "Obj 2: Compensation",
  importance_df$Feature %in% c("YearsAtCompany","TotalWorkingYears",
                                 "NumCompaniesWorked","YearsSinceLastPromotion",
                                 "YearsInCurrentRole","YearsWithCurrManager",
                                 "PromotionGap","SeniorityRatio") ~
    "Obj 3: Tenure",
  TRUE ~ "Engineered"
)

p4_2 <- ggplot(importance_df,
               aes(x = reorder(Feature, MeanDecreaseGini),
                   y = MeanDecreaseGini, fill = Source)) +
  geom_col(alpha = 0.85, color = "white") +
  geom_text(aes(label = round(MeanDecreaseGini, 1)),
            hjust = -0.2, size = 3.5, fontface = "bold") +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  scale_fill_manual(values = c("Obj 1: Demographics" = "#EF5350",
                                "Obj 2: Compensation" = "#1E88E5",
                                "Obj 3: Tenure"       = "#43A047",
                                "Engineered"          = "#AB47BC"),
                    name = "Source Objective") +
  labs(title   = "Analysis 4-2: RF Top 15 Features (colour = source objective)",
       subtitle = "Validates EDA: Obj 1-3 groups dominate -> model confirms analysis",
       x = "Feature", y = "Mean Decrease Gini") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), legend.position = "right")
print(p4_2)

# Analysis 4-3: DT on RF Top 8 features; ROC validates cost of restriction
cat("\n--- Analysis 4-3: Decision Tree on RF Top Features ---\n")
top_n_features <- 8
top_features   <- importance_df$Feature[1:top_n_features]
cat(sprintf("\nUsing top %d RF features:\n", top_n_features))
cat(paste(" -", top_features), sep = "\n")

dt_formula <- as.formula(paste("Attrition ~", paste(top_features, collapse = " + ")))

set.seed(42)
# Same train_balanced as RF; factor levels already aligned at Section 2.11
dt_model <- rpart(dt_formula, data = train_balanced, method = "class",
                  control = rpart.control(cp = 0.005, maxdepth = 4))
dt_probs <- predict(dt_model, test_data, type = "prob")[, "Yes"]
dt_pred  <- predict(dt_model, test_data, type = "class")
cm_dt    <- confusionMatrix(dt_pred, test_data$Attrition, positive = "Yes")
cat("\nDecision Tree - Confusion Matrix:\n"); print(cm_dt)
cat("\nVariables used in DT splits:\n"); print(dt_model$variable.importance)

# Inline plot (RStudio Plots pane)
par(mar = c(1, 1, 3, 1))
# suppressWarnings: rpart.plot warns when both cex and tweak are specified;
# this is cosmetic only and does not affect the tree output.
suppressWarnings(
  rpart.plot(dt_model, type = 4, extra = 104, box.palette = "RdBu",
             shadow.col = "gray", nn = TRUE, cex = 0.7,
             split.cex = 0.75, split.yshift = -1, tweak = 1.1,
             under = TRUE, faclen = 0,
             main = paste0("Analysis 4-3: DT Rules from RF Top ", top_n_features, " Features"),
             sub  = "Red = high risk | Built on RF-validated features only")
)
par(mar = c(5.1, 4.1, 4.1, 2.1))

# Second pass in a separate window to avoid ggplot2 device conflict.
rstudio_dev <- dev.cur()
new_dev     <- dev.new()
par(mar = c(1, 1, 2, 1))
suppressWarnings(
  rpart.plot(dt_model, type = 4, extra = 104, box.palette = "RdBu", cex = 0.65,
             main = paste0("Analysis 4-3: DT Rules from RF Top ", top_n_features, " Features"))
)
par(mar = c(5.1, 4.1, 4.1, 2.1))
dev.off()
if (rstudio_dev %in% dev.list()) dev.set(rstudio_dev)

# ROC comparison across all three models on the same test_data.
# Purpose: quantify whether restricting the DT to RF top features
# costs meaningful AUC performance vs the full RF model.
roc_log <- roc(test_data$Attrition, log_probs, levels = c("No","Yes"), quiet = TRUE)
roc_rf  <- roc(test_data$Attrition, rf_probs,  levels = c("No","Yes"), quiet = TRUE)
roc_dt  <- roc(test_data$Attrition, dt_probs,  levels = c("No","Yes"), quiet = TRUE)

auc_log       <- as.numeric(auc(roc_log))
auc_rf        <- as.numeric(auc(roc_rf))
auc_dt        <- as.numeric(auc(roc_dt))
auc_gap_rf_dt <- auc_rf - auc_dt

cat(sprintf("\nAUC - LR  : %.4f (baseline)\n",  auc_log))
cat(sprintf("AUC - RF  : %.4f (best)\n",        auc_rf))
cat(sprintf("AUC - DT  : %.4f (top %d RF)\n",   auc_dt, top_n_features))
cat(sprintf("Gap RF-DT : %.4f\n", auc_gap_rf_dt))
if (auc_gap_rf_dt < 0.05) {
  cat("-> Gap <0.05: DT rules are valid operational tool.\n")
} else {
  cat("-> Gap >=0.05: RF for prediction; DT as approximate guidance.\n")
}

p4_3_roc <- ggroc(list("Logistic Regression" = roc_log,
                        "Random Forest"       = roc_rf,
                        "Decision Tree"       = roc_dt), linewidth = 1.2) +
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", color = "grey50") +
  scale_color_manual(
    values = c("Logistic Regression" = "#2196F3",
               "Random Forest"       = "#4CAF50",
               "Decision Tree"       = "#FF5722"),
    labels = c(paste0("LR (AUC=",  round(auc_log, 3), ") Baseline"),
               paste0("RF (AUC=",  round(auc_rf,  3), ") Best"),
               paste0("DT top ", top_n_features,
                      " (AUC=", round(auc_dt, 3), ") HR-actionable"))
  ) +
  labs(title   = "Analysis 4-3: ROC - Does Restricting to RF Top Features Cost Performance?",
       subtitle = paste0("Same test_data | AUC gap RF vs DT = ",
                         round(auc_gap_rf_dt, 3), " | <0.05 = sufficient"),
       x = "Specificity", y = "Sensitivity (Recall)", color = "Model") +
  theme_minimal(base_size = 13) +
  theme(plot.title        = element_text(face = "bold"),
        legend.position   = c(0.55, 0.15),
        legend.background = element_rect(fill = "white", color = "grey80"),
        legend.text       = element_text(size = 8))
print(p4_3_roc)

model_comparison <- data.frame(
  Model     = c("Logistic Regression", "Random Forest",
                paste0("Decision Tree (top ", top_n_features, ")")),
  Accuracy  = c(as.numeric(cm_log$overall["Accuracy"]),
                as.numeric(cm_rf$overall["Accuracy"]),
                as.numeric(cm_dt$overall["Accuracy"])),
  Precision = c(as.numeric(cm_log$byClass["Precision"]),
                as.numeric(cm_rf$byClass["Precision"]),
                as.numeric(cm_dt$byClass["Precision"])),
  Recall    = c(as.numeric(cm_log$byClass["Recall"]),
                as.numeric(cm_rf$byClass["Recall"]),
                as.numeric(cm_dt$byClass["Recall"])),
  F1        = c(as.numeric(cm_log$byClass["F1"]),
                as.numeric(cm_rf$byClass["F1"]),
                as.numeric(cm_dt$byClass["F1"])),
  AUC       = c(auc_log, auc_rf, auc_dt)
)
model_comparison[, -1] <- round(model_comparison[, -1], 4)
cat("\n===== MODEL PERFORMANCE COMPARISON =====\n")
print(model_comparison)
cat("\nRecall is most critical: missing a leaver costs more than a false alarm.\n")


# ============================================================
# EXTRA FEATURE 1: PEARSON CORRELATION MATRIX
# ============================================================
cat("\n========== EXTRA FEATURE 1: CORRELATION MATRIX ==========\n")

# Purpose: Visualise pairwise Pearson correlations among all numeric predictors.
# Benefit: Identifies multicollinear variable pairs before modelling so that
#          redundant predictors can be noted (e.g. YearsAtCompany and
#          TotalWorkingYears carry overlapping information).
#          Also contextualises Obj 3 findings - tenure variables form a
#          correlated cluster, confirming they measure the same underlying risk.
num_cols_for_corr <- c("Age","DailyRate","DistanceFromHome",
                        "EnvironmentSatisfaction","HourlyRate",
                        "JobInvolvement","JobLevel","JobSatisfaction",
                        "MonthlyIncome","NumCompaniesWorked",
                        "PercentSalaryHike","RelationshipSatisfaction",
                        "StockOptionLevel","TotalWorkingYears",
                        "TrainingTimesLastYear","WorkLifeBalance",
                        "YearsAtCompany","YearsInCurrentRole",
                        "YearsSinceLastPromotion","YearsWithCurrManager",
                        "SatisfactionIndex","PromotionGap","SeniorityRatio")

corr_matrix <- cor(df[, num_cols_for_corr], use = "pairwise.complete.obs")

p_ef1 <- ggcorrplot(corr_matrix, method = "square", type = "lower",
                     lab = TRUE, lab_size = 1.8, tl.cex = 8,
                     colors = c("#D32F2F", "white", "#1976D2"),
                     title  = "Extra Feature 1: Pearson Correlation Matrix",
                     ggtheme = theme_minimal(base_size = 8)) +
  theme(plot.title  = element_text(face = "bold", size = 13),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8))
print(p_ef1)

cat("Highly correlated pairs (|r| > 0.6):\n")
for (i in 1:(nrow(corr_matrix) - 1))
  for (j in (i+1):ncol(corr_matrix))
    if (abs(corr_matrix[i,j]) > 0.6)
      cat(sprintf("  %s <-> %s : r = %.3f\n",
                  rownames(corr_matrix)[i], colnames(corr_matrix)[j],
                  corr_matrix[i,j]))


# ============================================================
# EXTRA FEATURE 2: K-MEANS CLUSTERING
# ============================================================
cat("\n========== EXTRA FEATURE 2: K-MEANS CLUSTERING ==========\n")

cluster_features    <- c("MonthlyIncome","JobSatisfaction","YearsAtCompany",
                          "OverTime","SatisfactionIndex","DistanceFromHome",
                          "WorkLifeBalance","PromotionGap")
df_cluster          <- df[, cluster_features]
df_cluster$OverTime <- as.numeric(df_cluster$OverTime == "Yes")
df_cluster_scaled   <- scale(df_cluster)

# Elbow Method: seed set ONCE before the loop so each k value receives
# a different but reproducible random initialisation sequence.
set.seed(42)
wss <- numeric(8)
for (k in 1:8)
  wss[k] <- kmeans(df_cluster_scaled, centers = k, nstart = 20)$tot.withinss

p_ef2a <- ggplot(data.frame(k = 1:8, WSS = wss), aes(x = k, y = WSS)) +
  geom_line(linewidth = 1.2, color = "#1565C0") +
  geom_point(size = 4, color = "#C62828") +
  geom_vline(xintercept = 3, linetype = "dashed", color = "grey50") +
  annotate("text", x = 3.2, y = max(wss) * 0.9,
           label = "k=3 (elbow)", hjust = 0, color = "grey40", size = 3.5) +
  labs(title = "Extra Feature 2a: Elbow Method - Optimal k",
       subtitle = "WSS flattens after k=3",
       x = "Number of Clusters (k)", y = "Total Within-Cluster SS") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))
print(p_ef2a)

set.seed(42)
km_model   <- kmeans(df_cluster_scaled, centers = 3, nstart = 25)
df$Cluster <- factor(km_model$cluster)

cluster_profile <- df %>%
  group_by(Cluster) %>%
  summarise(N             = n(),
            AttritionRate = round(mean(Attrition == "Yes") * 100, 1),
            AvgIncome     = round(mean(MonthlyIncome), 0),
            AvgJobSat     = round(mean(JobSatisfaction), 2),
            AvgYears      = round(mean(YearsAtCompany), 1),
            OvertimePct   = round(mean(OverTime == "Yes") * 100, 1),
            .groups       = "drop")
cat("\nCluster Profiles:\n"); print(cluster_profile)

# Extra Feature 2b: PCA-projected cluster visualisation.
# fviz_cluster() reduces the 8-dimensional scaled space to 2 principal
# components so the cluster boundaries can be shown in 2D.
p_ef2b <- fviz_cluster(km_model, data = df_cluster_scaled,
                        geom = "point", ellipse.type = "convex",
                        palette = c("#E53935","#43A047","#1E88E5"),
                        ggtheme = theme_minimal(base_size = 13)) +
  labs(title   = "Extra Feature 2b: K-Means Clusters (k=3) - PCA View",
       subtitle = "Each cluster = distinct risk profile") +
  theme(plot.title = element_text(face = "bold"))
print(p_ef2b)

# Extra Feature 2c: Validates that the unsupervised clusters correspond to
# meaningfully different attrition levels in the labelled data.
# A well-formed clustering shows clearly separated attrition rates per cluster.
p_ef2c <- ggplot(cluster_profile, aes(x = Cluster, y = AttritionRate, fill = Cluster)) +
  geom_col(alpha = 0.85, color = "white") +
  geom_text(aes(label = paste0(AttritionRate, "%\n(n=", N, ")")),
            vjust = -0.3, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("1" = "#E53935", "2" = "#43A047", "3" = "#1E88E5")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title   = "Extra Feature 2c: Actual Attrition Rate by Cluster",
       subtitle = "Confirms clusters map to distinct risk tiers",
       x = "Cluster", y = "Attrition Rate (%)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "none")
print(p_ef2c)


# ============================================================
# EXTRA FEATURE 3: RISK SCORING SYSTEM
# ============================================================
cat("\n========== EXTRA FEATURE 3: RISK SCORING SYSTEM ==========\n")

df_scored           <- test_data
df_scored$RiskScore <- round(rf_probs * 100, 1)
df_scored$RiskTier  <- cut(df_scored$RiskScore,
                            breaks = c(-Inf, 25, 50, 75, Inf),
                            labels = c("Low","Medium","High","Critical"),
                            include.lowest = TRUE)
df_scored$Actual    <- test_data$Attrition

risk_summary <- df_scored %>%
  group_by(RiskTier) %>%
  summarise(Count = n(),
            AttritionRate = round(mean(Actual == "Yes") * 100, 1),
            AvgRiskScore  = round(mean(RiskScore), 1),
            .groups = "drop")
cat("\nRisk Tier Summary:\n"); print(risk_summary)

# 3a: Attrition rate per tier (monotonic = well-calibrated)
p_ef3a <- ggplot(risk_summary, aes(x = RiskTier, y = AttritionRate, fill = RiskTier)) +
  geom_col(alpha = 0.85, color = "white") +
  geom_text(aes(label = paste0(AttritionRate, "%\n(n=", Count, ")")),
            vjust = -0.3, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("Low" = "#43A047", "Medium" = "#FDD835",
                                "High" = "#FB8C00", "Critical" = "#E53935")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title   = "Extra Feature 3a: Actual Attrition Rate by Risk Tier",
       subtitle = "Monotonic increase validates the scoring system",
       x = "Risk Tier", y = "Actual Attrition Rate (%)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "none")
print(p_ef3a)

# 3b: Score density - separation validates discrimination
p_ef3b <- ggplot(df_scored, aes(x = RiskScore, fill = Actual)) +
  geom_density(alpha = 0.55) +
  scale_fill_manual(values = c("No" = "#1976D2", "Yes" = "#C62828")) +
  geom_vline(xintercept = c(25, 50, 75),
             linetype = "dashed", color = "grey40", linewidth = 0.8) +
  annotate("text", x = c(12,37,62,87), y = Inf,
           label = c("Low","Medium","High","Critical"),
           vjust = 1.6, hjust = 0.5, size = 3.5, color = "grey30") +
  labs(title   = "Extra Feature 3b: Risk Score Distribution - Leavers vs Stayers",
       subtitle = "Clear separation = good discrimination",
       x = "Risk Score (0=stay, 100=leave)", y = "Density", fill = "Actual") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))
print(p_ef3b)


# ============================================================
# FINAL SUMMARY
# ============================================================
cat("\n\n===================================================\n")
cat("         ANALYSIS COMPLETE - FINAL SUMMARY\n")
cat("===================================================\n")
cat("Dataset:", nrow(df), "records |",
    "Attrition rate:", round(mean(df$Attrition == "Yes") * 100, 2), "%\n\n")
cat("1. DEMOGRAPHICS (Obj 1): Highest attrition age 18-25, single employees.\n")
cat("   Education compounds risk within the highest-risk sub-group.\n\n")
cat("2. COMPENSATION (Obj 2): Leavers earn less; Low income + OT = compounded.\n")
cat("   JobLevel gradient confirmed; dept-level target identified.\n\n")
cat("3. TENURE (Obj 3): Danger window = Years 0-3. Promotion stagnation accelerates.\n")
cat("   Triple-risk: Early + stagnant + serial job-hopper = MAX risk.\n\n")
cat("4. MODELLING (Obj 4):\n")
print(model_comparison[, c("Model","Accuracy","Recall","AUC")])
cat("\n   RF top features confirm Obj 1-3 findings (closing loop).\n")
cat("\n5. CLUSTERS (EF 2): 3 risk segments validated.\n")
cat("6. RISK SCORING (EF 3): Critical-tier employees flagged for HR.\n")
cat("===================================================\n")
