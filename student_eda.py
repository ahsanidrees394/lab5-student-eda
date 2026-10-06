import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Create student dataset
# --------------------------------------------------

data = {
    "StudentID": [
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11
    ],

    "Gender": [
        "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male",
        "Female", "Male", "Female"
    ],

    "StudyHours": [
        4, 3, 5, 2, 6, 4,
        7, 3, 5, 2, 6
    ],

    "Attendance(%)": [
        90, 82, 92, 78, 95, 85,
        88, 80, 91, 76, 94
    ],

    "Marks": [
        82, 70, 88, 65, np.nan, 74,
        91, 68, 84, np.nan, 89
    ]
}

df = pd.DataFrame(data)

# Add one exact duplicate row.
# This makes the CSV contain 12 rows total.
df = pd.concat(
    [df, df.iloc[[0]]],
    ignore_index=True
)

# Save dataset
df.to_csv("student_performance.csv", index=False)

# --------------------------------------------------
# Load CSV using pd.read_csv()
# --------------------------------------------------

df = pd.read_csv("student_performance.csv")

# --------------------------------------------------
# A. Shape and info
# --------------------------------------------------

print("Shape:", df.shape)

print("\nInfo:")
df.info()

# --------------------------------------------------
# B. Missing values and duplicates
# --------------------------------------------------

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# --------------------------------------------------
# Remove duplicate
# --------------------------------------------------

df_clean = df.drop_duplicates().copy()

# --------------------------------------------------
# Fill missing Marks with median
# --------------------------------------------------

median_marks = df_clean["Marks"].median()

df_clean["Marks"] = df_clean["Marks"].fillna(
    median_marks
)

print("\nAfter cleaning:", df_clean.shape)

print(
    "Median used for missing Marks:",
    median_marks
)

# --------------------------------------------------
# C. Histogram of Marks
# --------------------------------------------------

plt.figure(figsize=(7, 4))

plt.hist(
    df_clean["Marks"],
    bins=6,
    edgecolor="black"
)

plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.title("Marks Distribution")

plt.tight_layout()

plt.savefig(
    "marks_histogram.png",
    dpi=150
)

plt.show()

# --------------------------------------------------
# Boxplot of Marks
# --------------------------------------------------

plt.figure(figsize=(6, 4))

plt.boxplot(df_clean["Marks"])

plt.ylabel("Marks")
plt.title("Marks Boxplot")

plt.tight_layout()

plt.savefig(
    "marks_boxplot.png",
    dpi=150
)

plt.show()

# --------------------------------------------------
# D. Scatter plot
# --------------------------------------------------

plt.figure(figsize=(7, 4))

plt.scatter(
    df_clean["StudyHours"],
    df_clean["Marks"]
)

plt.xlabel("Study Hours")
plt.ylabel("Marks")

plt.title(
    "Study Hours vs Marks"
)

plt.tight_layout()

plt.savefig(
    "studyhours_vs_marks.png",
    dpi=150
)

plt.show()

# --------------------------------------------------
# Correlation matrix
# --------------------------------------------------

numeric_columns = [
    "StudyHours",
    "Attendance(%)",
    "Marks"
]

correlation_matrix = (
    df_clean[numeric_columns].corr()
)

print("\nCorrelation matrix:")
print(correlation_matrix)

# --------------------------------------------------
# E. Group by Gender
# --------------------------------------------------

gender_summary = (
    df_clean
    .groupby("Gender")[[
        "Marks",
        "Attendance(%)"
    ]]
    .mean()
)

print(
    "\nAverage Marks and Attendance by Gender:"
)

print(gender_summary)

print(
    "\nSaved marks_histogram.png, "
    "marks_boxplot.png, "
    "studyhours_vs_marks.png"
)