"""
Student Performance and Study-Hours Dashboard
Run: python dashboard.py
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "students.csv"


def score_percentage(obtained, total):
    """Convert marks to percentage."""
    if total <= 0:
        return 0
    return (obtained / total) * 100


def get_grade(score):
    """Return a simple letter grade."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


class StudentRecord:
    def __init__(self, student_id, course, study_hours, score):
        self.student_id = student_id
        self.course = course
        self.study_hours = study_hours
        self.score = score

    def grade(self):
        return get_grade(self.score)


def load_students(filename):
    """Read CSV records into dictionaries."""
    students = []
    with open(filename, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            students.append({
                "student_id": row["student_id"],
                "course": row["course"],
                "study_hours": float(row["study_hours"]),
                "score": float(row["score"])
            })
    return students


def main():
    students = load_students(CSV_FILE)

    # Create StudentRecord objects
    records = [
        StudentRecord(s["student_id"], s["course"], s["study_hours"], s["score"])
        for s in students
    ]

    # NumPy: average and spread of scores
    scores = np.array([s["score"] for s in students])
    print("Number of students:", len(scores))
    print("Average score:", round(np.mean(scores), 2))
    print("Score standard deviation:", round(np.std(scores), 2))

    # Display grades
    print("\\nStudent grades:")
    for record in records:
        print(record.student_id, record.course, record.score, record.grade())

    # Pandas: course-wise summary
    df = pd.DataFrame(students)
    print("\\nCourse-wise average score:")
    print(df.groupby("course")["score"].mean().round(2))

    # Scatter plot: study hours vs score
    plt.figure()
    plt.scatter(df["study_hours"], df["score"])
    plt.xlabel("Study Hours")
    plt.ylabel("Assessment Score")
    plt.title("Study Hours vs Assessment Score")
    plt.grid(True)
    plt.savefig("study_hours_vs_score.png")
    plt.show()

    # Histogram: score distribution
    plt.figure()
    plt.hist(df["score"], bins=8, edgecolor="black")
    plt.xlabel("Assessment Score")
    plt.ylabel("Number of Students")
    plt.title("Score Distribution")
    plt.savefig("score_histogram.png")
    plt.show()

    # Simple correlation and outlier check
    correlation = df["study_hours"].corr(df["score"])
    print("\\nCorrelation (study hours and score):", round(correlation, 3))

    # Remove unusually high/low scores using the IQR rule
    q1 = df["score"].quantile(0.25)
    q3 = df["score"].quantile(0.75)
    iqr = q3 - q1
    filtered = df[(df["score"] >= q1 - 1.5 * iqr) &
                  (df["score"] <= q3 + 1.5 * iqr)]

    filtered_corr = filtered["study_hours"].corr(filtered["score"])
    print("Correlation after score outlier check:", round(filtered_corr, 3))

    print("\\nEvaluation:")
    print("Correlation shows association, not causation.")
    print("Other factors may affect scores, such as prior knowledge, sleep,")
    print("teaching quality, and study methods.")
    print("Compare the correlations before and after outlier filtering.")


if __name__ == "__main__":
    main()
