# ===========================
# Task 1: Load and Explore the Dataset
# ===========================

import pandas as pd            # Import pandas for data handling and analysis
import matplotlib.pyplot as plt # Import matplotlib for plotting/visualization
import seaborn as sns          # Import seaborn for sample datasets and easy plotting

# Optional: Show plots inside the Jupyter Notebook itself
%matplotlib inline

# ---------------------------
# Step 1: Load the dataset
# ---------------------------
try:
    # Option 1: Load CSV file from local storage
    # Uncomment the line below and replace 'your_dataset.csv' with your file name if using a local CSV file.
    # df = pd.read_csv("your_dataset.csv")

    # Option 2: Load a sample dataset provided by seaborn (here, we use the 'iris' dataset)
    # This dataset contains measurements of iris flowers (sepal length, sepal width, petal length, petal width)
    df = sns.load_dataset("iris")
    print("✅ Dataset loaded successfully!")

except FileNotFoundError:
    # This will run if the file you try to load does not exist in the folder
    print("❌ Error: File not found. Check the file path and name.")
except Exception as e:
    # This will run if there is any other error when loading the dataset
    print(f"❌ Error loading dataset: {e}")

# ---------------------------
# Step 2: Inspect the dataset
# ---------------------------

# Display the first 5 rows of the dataset so we can see what the data looks like
print("\nFirst 5 rows of the dataset:")
display(df.head())

# Show information about the dataset including column names, data types, and non-null counts
print("\nDataset Info:")
print(df.info())

# Show how many missing (null) values are in each column
print("\nMissing values per column:")
print(df.isnull().sum())

# ---------------------------
# Step 3: Clean the dataset
# ---------------------------

# Drop any rows that contain missing values (if there are any)
df_clean = df.dropna()

# Print the shape (rows, columns) of the cleaned dataset
print(f"\n✅ Cleaned dataset shape: {df_clean.shape}")

# ===========================
# Task 2: Basic Data Analysis
# ===========================

# ---------------------------
# Step 1: Basic statistics
# ---------------------------
# Generate summary statistics for all numeric columns in the cleaned dataset.
# This includes count, mean, standard deviation, min, max, and quartile values.
print("\nDescriptive statistics:")
display(df_clean.describe())

# ---------------------------
# Step 2: Grouping by Category
# ---------------------------
# Group the data by the 'species' column and calculate the mean for each numeric column.
# This shows the average sepal length, sepal width, petal length, and petal width per species.
print("\nAverage measurements per species:")
grouped = df_clean.groupby("species").mean()
display(grouped)

# ---------------------------
# Step 3: Identify patterns
# ---------------------------
# Print some observations based on the dataset.
# These are insights we can see by comparing the average values from the grouped data.
print("\nInteresting findings:")
print("- Setosa species has the smallest petal length on average.")
print("- Virginica species has the largest petal length and width.")

# ===========================
# Task 3: Data Visualization
# ===========================

# Set a clean and nice visual style using seaborn
plt.style.use("seaborn-v0_8")

# -------------------------------------------------
# 1️⃣ Line Chart (Trends over index - not time-based)
# -------------------------------------------------
# Plotting the sepal length values across the dataset index.
# Although the index is not a time series, this helps us visualize the variation of sepal length.
plt.figure(figsize=(8,4))
plt.plot(df_clean.index, df_clean["sepal_length"], label="Sepal Length")
plt.title("Line Chart of Sepal Length")  # Title of the chart
plt.xlabel("Index")                      # X-axis label
plt.ylabel("Sepal Length")               # Y-axis label
plt.legend()                             # Show legend for clarity
plt.show()

# -------------------------------------------------
# 2️⃣ Bar Chart (Mean petal length per species)
# -------------------------------------------------
# Using seaborn's barplot to show the average petal length for each species.
# This makes it easy to compare species visually.
plt.figure(figsize=(6,4))
sns.barplot(x="species", y="petal_length", data=df_clean, estimator="mean")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Mean Petal Length")
plt.show()

# -------------------------------------------------
# 3️⃣ Histogram (Distribution of sepal length)
# -------------------------------------------------
# Plotting a histogram to show the frequency distribution of sepal length.
# The bins parameter defines how many groups to divide the data into.
plt.figure(figsize=(6,4))
plt.hist(df_clean["sepal_length"], bins=15, edgecolor="black")
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# -------------------------------------------------
# 4️⃣ Scatter Plot (Sepal length vs Petal length)
# -------------------------------------------------
# Scatter plot helps visualize the relationship between sepal length and petal length.
# The hue parameter colors the points based on species, making it easy to see clusters.
plt.figure(figsize=(6,4))
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df_clean)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()  # Show species legend
plt.show()
