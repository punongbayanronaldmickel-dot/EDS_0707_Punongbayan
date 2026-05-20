import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

# =========================================
# FOLDERS
# =========================================
os.makedirs("outputs", exist_ok=True)

# =========================================
# AUTO-DETECT DATASET FILE
# =========================================
def load_dataset():
    try:
        # Automatically find CSV inside data folder
        files = os.listdir("data")
        csv_files = [f for f in files if f.endswith(".csv")]

        if len(csv_files) == 0:
            print("ERROR: No CSV file found inside data folder.")
            exit()

        file_path = os.path.join("data", csv_files[0])
        print(f"Loading dataset: {file_path}\n")

        df = pd.read_csv(file_path)
        return df

    except Exception as e:
        print("ERROR loading dataset:", e)
        exit()

# =========================================
# CLEANING
# =========================================
def clean_dataset(df):

    print("\nMissing values BEFORE cleaning:\n")
    print(df.isnull().sum())

    if "Occlusion_Level" in df.columns:
        df["Occlusion_Level"] = df["Occlusion_Level"].fillna("Unknown")

    df = df.drop_duplicates()

    df.to_csv("data/dataset_cleaned.csv", index=False)

    print("\nDataset cleaned and saved.\n")

    return df

# =========================================
# STATISTICS
# =========================================
def statistical_analysis(df):

    print("\n========== DESCRIPTIVE STATISTICS ==========")

    length = df["Defect_Length_mm"].to_numpy()
    width = df["Defect_Width_mm"].to_numpy()
    depth = df["Defect_Depth_mm"].to_numpy()

    print("MEAN LENGTH:", np.mean(length))
    print("MEDIAN DEPTH:", np.median(depth))
    print("STD WIDTH:", np.std(width))
    print("VAR LENGTH:", np.var(length))

    print("\n========== GROUP ANALYSIS ==========")

    print(
        df.groupby("Infrastructure_Type")[
            ["Defect_Length_mm", "Defect_Depth_mm"]
        ].mean()
    )

    print("\n========== CORRELATION ==========")

    corr = df[[
        "Defect_Length_mm",
        "Defect_Width_mm",
        "Defect_Depth_mm"
    ]].corr()

    print(corr)

    return corr

# =========================================
# STATIC VISUALS
# =========================================
def create_static_graphs(df, corr):

    # Heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Defect Parameter Correlation")
    plt.savefig("outputs/heatmap.png")
    plt.show()

    # Histogram
    plt.figure(figsize=(7,5))
    plt.hist(df["Defect_Depth_mm"], bins=30)
    plt.title("Distribution of Defect Depth")
    plt.xlabel("Depth (mm)")
    plt.ylabel("Frequency")
    plt.savefig("outputs/histogram.png")
    plt.show()

    # Boxplot
    plt.figure(figsize=(7,5))
    sns.boxplot(x="Severity_Level", y="Defect_Length_mm", data=df)
    plt.title("Severity vs Defect Length")
    plt.savefig("outputs/boxplot.png")
    plt.show()

# =========================================
# ANIMATIONS
# =========================================
def create_animations(df):

    # ensure no NaN issues
    df = df.copy()
    df["Inspection_Mode"] = df["Inspection_Mode"].astype(str)

    # 1. Infrastructure distribution
    fig1 = px.box(
        df,
        x="Infrastructure_Type",
        y="Defect_Length_mm",
        color="Infrastructure_Type",
        title="Defect Length Distribution per Infrastructure Type"
    )
    fig1.write_html("outputs/animation_infrastructure.html")
    fig1.show()

    # 2. Severity behavior
    fig2 = px.scatter(
        df,
        x="Defect_Length_mm",
        y="Defect_Depth_mm",
        color="Severity_Level",
        animation_frame="Inspection_Mode",
        title="Defect Severity Behavior Across Inspection Modes"
    )
    fig2.write_html("outputs/animation_severity.html")
    fig2.show()

    # 3. Intensity
    fig3 = px.bar(
        df,
        x="Infrastructure_Type",
        y="Defect_Length_mm",
        color="Severity_Level",
        animation_frame="Inspection_Mode",
        title="Defect Intensity Across Inspection Modes"
    )
    fig3.write_html("outputs/animation_intensity.html")
    fig3.show()

# =========================================
# MAIN
# =========================================
def main():

    df = load_dataset()

    print("\nPREVIEW:")
    print(df.head())

    print("\nCOLUMNS:")
    print(df.columns)

    print("\nINFO:")
    print(df.info())

    df = clean_dataset(df)

    corr = statistical_analysis(df)

    create_static_graphs(df, corr)

    create_animations(df)

    print("\nPROJECT EXECUTION COMPLETE.")

# RUN
if __name__ == "__main__":
    main()