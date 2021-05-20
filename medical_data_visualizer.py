import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import data
df = pd.read_csv("./medical_examination.csv")

# Add 'overweight' column


def get_overweight_val(column):
    bmi = column["weight"] / (column["height"] / 100) ** 2
    # bmi = column['weight'] / (2 ** column['height'] / 100)
    return 1 if bmi > 25 else 0


df["overweight"] = df.apply(get_overweight_val, axis=1)
# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df["gluc"] = df["gluc"].apply(lambda val: 0 if val == 1 else 1)
df["cholesterol"] = df["cholesterol"].apply(lambda val: 0 if val == 1 else 1)

# Draw Categorical Plot


def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke",
                    "alco", "active", "overweight"],
    )

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat["total"] = 1
    df_cat = df_cat.groupby(
        ["value", "cardio", "variable"], as_index=False).count()

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(
        data=df_cat, kind="bar", col="cardio", hue="value", x="variable", y="total"
    ).fig

    # Do not modify the next two lines
    fig.savefig("catplot.png")
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.loc[
        (df["ap_lo"] <= df["ap_hi"])
        & (df["height"] >= df["height"].quantile(0.025))
        & (df["height"] <= df["height"].quantile(0.975))
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with 'sns.heatmap()'
    ax = sns.heatmap(corr, mask=mask, square=True,
                     annot=True, fmt=".1f", ax=ax)

    # Do not modify the next two lines
    fig.savefig("heatmap.png")
    return fig
