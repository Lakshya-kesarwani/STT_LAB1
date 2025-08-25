import os
import pandas as pd
from pydriller import Repository
from git import Repo

def clean_diff(diff_str):
    """Clean diff by ignoring metadata, whitespace, and blank lines."""
    lines = diff_str.split('\n')
    cleaned_lines = []
    for line in lines:
        if not line.strip():
            continue
        if line.startswith(('---', '+++', '@@', 'diff --git')):
            continue
        # Remove leading '+' or '-' only (diff markers), keep actual code
        if line[0] in ['+', '-']:
            cleaned_lines.append(line[1:].strip())
        else:
            cleaned_lines.append(line.strip())
    return " ".join(cleaned_lines)

# Example: dictionary of repos
selected_repos = {"JAX":"https://github.com/jax-ml/jax","SKIA":"https://github.com/google/skia","LLAMA":"https://github.com/meta-llama/llama3"}


for repo_name, repo_url in selected_repos.items():
    print(f"--- Analyzing repository: {repo_name} ---")

    repo_data = []
    modified_file_count = 0
    file_limit = 1000

    repo_path = f"./{repo_name}"
    if not os.path.exists(repo_path):
        Repo.clone_from(repo_url, repo_path)
    repo = Repo(repo_path)

    try:
        for commit in Repository(repo_path).traverse_commits():
            if modified_file_count >= file_limit:
                break

            parent_sha = commit.parents[0] if commit.parents else None
            if parent_sha is None:
                continue
            parent_commit = repo.commit(parent_sha)

            for mod in commit.modified_files:
                if modified_file_count >= file_limit:
                    break

                if mod.old_path is None or mod.new_path is None:
                    continue

                diff_myers1 = repo.git.diff(parent_commit.hexsha, commit.hash,
                                            "--diff-algorithm=myers",
                                            "--", mod.new_path)
                diff_hist2 = repo.git.diff(parent_commit.hexsha, commit.hash,
                                           "--diff-algorithm=histogram",
                                           "--", mod.new_path)

                diff_myers1_cleaned = clean_diff(diff_myers1)
                diff_hist2_cleaned = clean_diff(diff_hist2)

                discrepancy = "Yes" if diff_myers1_cleaned != diff_hist2_cleaned else "No"

                file_data = {
                    'old_filepath': mod.old_path,
                    'new_filepath': mod.new_path,
                    'commitSHA': commit.hash,
                    'parentcommitSHA': parent_sha,
                    'commit_message': commit.msg,
                    'diff_myers1': diff_myers1,
                    'diff_hist2': diff_hist2,
                    'Discrepancy': discrepancy
                }
                print(f"{modified_file_count} : Data added for {mod.new_path} in repo {repo_name}")
                repo_data.append(file_data)
                modified_file_count += 1

    except Exception as e:
        print(f"An error occurred while analyzing {repo_name}: {e}")
        continue

    if repo_data:
        df = pd.DataFrame(repo_data)
        output_filename = f"data/{repo_name.lower()}_data.csv"
        df.to_csv(output_filename, index=False, encoding="utf-8", errors="replace")
        print(f"Successfully generated {output_filename}")
    else:
        print(f"No data was collected for {repo_name}.")
