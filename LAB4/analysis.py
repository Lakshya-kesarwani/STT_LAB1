import pandas as pd
import sys

def analyze_discrepancies(csv_file):
    df = pd.read_csv(csv_file)

    if 'Discrepancy' not in df.columns:
        print(f"[ERROR] {csv_file} does not contain a 'Discrepancy' column.")
        return None

    total_files = len(df)
    discrepancy_counts = df['Discrepancy'].value_counts()

    yes_count = discrepancy_counts.get("Yes", 0)
    no_count = discrepancy_counts.get("No", 0)

    yes_percent = (yes_count / total_files * 100) if total_files > 0 else 0
    no_percent = (no_count / total_files * 100) if total_files > 0 else 0

    print("="*60)
    print(f"Analysis of: {csv_file}")
    print(f"Total files analyzed: {total_files}")
    print(f"Discrepancy 'Yes': {yes_count} ({yes_percent:.2f}%)")
    print(f"Discrepancy 'No' : {no_count} ({no_percent:.2f}%)")
    print("="*60)

    return {"repo": csv_file, "total": total_files, "yes": yes_count, "no": no_count}


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python analysis.py <csv_file1> <csv_file2> ...")
#     else:
files = ['jax_data.csv', 'skia_data.csv', 'llama_data.csv']
results = []
for csv_file in files:
    res = analyze_discrepancies("data/"+csv_file)
    if res:
        results.append(res)

# Combined summary
if results:
    total_files_all = sum(r["total"] for r in results)
    total_yes = sum(r["yes"] for r in results)
    total_no = sum(r["no"] for r in results)

    yes_percent_all = (total_yes / total_files_all * 100) if total_files_all > 0 else 0
    no_percent_all = (total_no / total_files_all * 100) if total_files_all > 0 else 0

    print("\n📊 Combined Summary Across Repos")
    print("="*60)
    for r in results:
        print(f"{r['repo']}: {r['yes']} Yes ({r['yes']/r['total']*100:.2f}%), "
                f"{r['no']} No ({r['no']/r['total']*100:.2f}%), "
                f"Total={r['total']}")
    print("="*60)
    print(f"TOTAL: {total_files_all}")
    print(f"Discrepancy 'Yes': {total_yes} ({yes_percent_all:.2f}%)")
    print(f"Discrepancy 'No' : {total_no} ({no_percent_all:.2f}%)")
    print("="*60)
    with open("data/summary.txt", "w") as f:
        f.write("📊 Combined Summary Across Repos\n")
        f.write("="*60 + "\n")
        for r in results:
            f.write(f"{r['repo']}: {r['yes']} Yes ({r['yes']/r['total']*100:.2f}%), "
                    f"{r['no']} No ({r['no']/r['total']*100:.2f}%), "
                    f"Total={r['total']}\n")
        f.write("="*60 + "\n")
        f.write(f"TOTAL: {total_files_all}\n")
        f.write(f"Discrepancy 'Yes': {total_yes} ({yes_percent_all:.2f}%)\n")
        f.write(f"Discrepancy 'No' : {total_no} ({no_percent_all:.2f}%)\n")
        f.write("="*60 + "\n")