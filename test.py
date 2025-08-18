import difflib
import ast
from pydriller import Repository, Commit
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- PARAMETERS ---
repo_path = "https://github.com/Naveen-Pal/SRIP"  # <-- Clone your repo locally first
bugs = ["bug", "fix", "issue", "error", "problem", "fail", "exception", "crash", "fault", "defect", 'refactor', 'resolved']
max_limit = 1000

# --- STEP 1: Collect bug-related commits ---
data = {'Hash': [], 'Message': [], 'Hashes of parents': [], 'Is a merge commit?': [], 'List of modified files': []}

for i, commit in enumerate(Repository(repo_path).traverse_commits()):
    if i >= max_limit:
        break
    message_words = [word.lower() for word in commit.msg.split()]
    if any(keyword in message_words for keyword in bugs):
        data['Hash'].append(commit.hash)
        # Join parent hashes with comma
        parent_hashes_str = ",".join(commit.parents) if commit.parents else ""
        data['Hashes of parents'].append(parent_hashes_str)
        data['Message'].append(f"{commit.author.name}: {commit.msg}")
        data['Is a merge commit?'].append(commit.merge)
        data['List of modified files'].append([f.filename for f in commit.modified_files])

df_commits = pd.DataFrame(data)
df_commits.to_csv('commit_data.csv', index=False)

# --- Helper functions ---

def get_diff(before, current):
    return "\n".join(difflib.unified_diff(
        before.splitlines() if before else [],
        current.splitlines() if current else [],
        lineterm=''
    ))

def llm_inference(tokenizer, model, diff):
    inputs = tokenizer(diff, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def rectifier(tokenizer, model, message, before, current):
    template = f"""
You are a commit message rectifier. Your task is to improve the commit message by making it more descriptive and clear.
Here is the before code: {before}
Here is the current code: {current}
And this was the original message: {message}
"""
    inputs = tokenizer(template, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_file_content(filename, commit):
    if commit is None:
        return None
    for modified_file in commit.modified_files:
        if modified_file.filename == filename:
            return modified_file.source_code
    return None

# --- STEP 2: Process each commit for file diffs and LLM inference ---

tokenizer = AutoTokenizer.from_pretrained("mamiksik/CommitPredictorT5")
model = AutoModelForSeq2SeqLM.from_pretrained("mamiksik/CommitPredictorT5")

rows = []

for _, row in df_commits.iterrows():
    files = row['List of modified files']
    if isinstance(files, str):
        import ast
        try:
            files = ast.literal_eval(files)
        except Exception as e:
            print(f"Error parsing modified files list: {e}")
            files = []
    elif isinstance(files, list):
        pass  # already good
    else:
        files = []


    for file in files:
        # Relax alnum check: skip if empty or not a string
        if not isinstance(file, str) or not file.strip():
            continue

        commit_hash = row['Hash']
        try:
            from pydriller.git import Git
            git_repo = Git(repo_path)
            commit = git_repo.get_commit(commit_hash)
        except StopIteration:
            print(f"Commit {commit_hash} not found")
            continue

        # Get parent commit object (only first parent)
        parent_hashes = row['Hashes of parents'].split(",") if row['Hashes of parents'] else []
        if parent_hashes:
            parent_commit_hash = parent_hashes[0]
            try:
                parent_commit = None
                if parent_hashes:
                    parent_commit = git_repo.get_commit(parent_hashes[0])
            except StopIteration:
                parent_commit = None
        else:
            parent_commit = None

        before_code = get_file_content(file, parent_commit)
        current_code = get_file_content(file, commit)
        diff_text = get_diff(before_code, current_code)
        llm_message = llm_inference(tokenizer, model, diff_text)
        rectified_msg = rectifier(tokenizer, model, commit.msg, before_code, current_code)

        rows.append({
            "Hash": commit.hash,
            "Message": commit.msg,
            "Filename": file,
            "Source Code (before)": before_code,
            "Source Code (current)": current_code,
            "Diff": diff_text,
            "LLM Inference (fix type)": llm_message,
            "Rectified Message": rectified_msg
        })

df_results = pd.DataFrame(rows)
print(df_results.head())
print(f"Found {df_results.shape[0]} commits with bug-related messages.")
df_results.to_csv("Bug_Mining.csv", index=False)
print("Data saved to Bug_Mining.csv")
