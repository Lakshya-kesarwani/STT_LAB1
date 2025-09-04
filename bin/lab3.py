import pandas as pd

df = pd.read_excel('STT_LAB2_BUG_MINING.xlsx')
print(df.head())



# Load model directly
# from transformers import AutoTokenizer, AutoModel

# # tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# # model = AutoModel.from_pretrained("microsoft/codebert-base")

print("--- Baseline Descriptive Statistics ---")

total_commits = df['Hash'].nunique()
total_files = df['Filename'].nunique()
print("Q1: Total number of commits and files.")

print(f"\nTotal number of unique commits: {total_commits}")
print(f"Total number of unique files: {total_files}")


files_per_commit = df.groupby('Hash')['Filename'].count()
avg_files_per_commit = files_per_commit.mean()
print("Q2: Average number of modified files per commit.")
print(f"\nAverage number of modified files per commit: {avg_files_per_commit:.2f}")


fix_type_distribution = df['LLM Inference (fix type)'].value_counts()

print("Q3: Distribution of fix types from LLM Inference (fix type).")
print("\nDistribution of fix types:")
print(fix_type_distribution)


top_modified_files = df['Filename'].value_counts().head(5) # Get top 5

print("Q4: Most frequently modified filenames/extensions.")
print("\nTop 5 most frequently modified files:")
print(top_modified_files)

# Now, let's extract the extension and find the most common ones.
# We use .str.split('.').str[-1] to get the last part after a dot.
df['Extension'] = df['Filename'].str.split('.').str[-1]
top_extensions = df['Extension'].value_counts()

# PLOT
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
top_modified_files.plot(kind='bar', color='skyblue')
plt.title("Top 5 Most Frequently Modified Files")
plt.xlabel("Files")
plt.ylabel("Number of Edits")

plt.subplot(1, 2, 2)
top_extensions.plot(kind='bar', color='salmon')
plt.title("Frequency of Modified File Extensions")
plt.xlabel("File Extensions")
plt.ylabel("Number of Edits")

plt.tight_layout()
plt.show()

print("\nFrequency of modified file extensions:")
print(top_extensions)