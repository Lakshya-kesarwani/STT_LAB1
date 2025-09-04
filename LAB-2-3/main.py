import difflib
from fileinput import filename
from unittest import case
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydriller import Repository
import pandas as pd
data = {'Hash':[] ,'Message':[] ,'Hashes of parents':[] ,'Is a merge commit?':[] ,'List of modified files':[]}


urls = ["https://github.com/jax-ml/jax"]
bugs = ["bug", "fix", "issue", "error", "problem", "fail", "exception", "crash", "fault", "defect",'refactor','resolved']
max_limit = 20
for i,commit in enumerate(Repository(path_to_repo=urls).traverse_commits()):
    if i >= max_limit:
        break
    message = [msg.lower() for msg in commit.msg.split(' ')]
    for i in bugs:
        if i in message:
            data['Hash'].append(commit.hash)
            data['Hashes of parents'].append(commit.parents)
            data['Message'].append(commit.author.name + ": " + commit.msg)
            data['Is a merge commit?'].append(commit.merge)
            data['List of modified files'].append([f.filename for f in commit.modified_files])
            break
# For each modified file (in the previous step), store the following (as csv).
data = pd.DataFrame(data).to_csv('commit_data.csv', index=False)
"""Hash, Message, Filename, Source Code (before), Source Code (current), Diff, LLM Inference (fix type), Rectified Message"""
from pydriller import Commit
def get_diff(before,current):
    diff = []
    for line in difflib.unified_diff(before.splitlines(), current.splitlines(), lineterm=''):
        diff.append(line)
    return '\n'.join(diff)

def llm_inference(model, diff):
    # Call the LLM model with the diff and return the response
    response = model.predict(diff)
    return response
def rectifier(message,before,current):
    # Call the LLM model with the message and return the rectified response
    template = f"""
    You are a commit message rectifier. Your task is to improve the commit message by making it more descriptive and clear.
    here is the before code: {before},
    here is the current code: {current},
    And this was the original message: {message}
    """
    try :
        response = model.predict(template)
    except Exception as e:
        print(f"Error occurred: {e}")
        response = "Error in rectification"
    return response
def get_file_content(filename, commit):
    # Get the content of the file at the specified commit
    for modified_file in commit.modified_files:
        if modified_file.filename == filename:
            return modified_file.source_code
    return None

df = pd.read_csv('commit_data.csv')
tokenizer = AutoTokenizer.from_pretrained("mamiksik/CommitPredictorT5")
model = AutoModelForSeq2SeqLM.from_pretrained("mamiksik/CommitPredictorT5")
for index, row in df.iterrows():

   
    filename = row['List of modified files']

    for file in filename:
        if( type(file) is not str or not file.isalnum()):
            continue
        hash = row['Hash']
        msg = row['Message']
        commit = Commit(hash)
        before = get_file_content(file, commit.parents[0])
        current = get_file_content(file, commit)
        diff = get_diff(before, current)
        llm_message = llm_inference(model, diff)
        rectified = rectifier(commit.msg, before, current)

        # Append the information to the DataFrame
        df = df.append({
            "Hash": commit.hash,
            "Message": commit.msg,
            "Filename": file,
            "Source Code (before)": before,
            "Source Code (current)": current,
            "Diff ": diff,
            "LLM Inference (fix type)": llm_message,
            "Rectified Message": rectified
        }, ignore_index=True)

df = pd.DataFrame(data)
print(df.head())
print(f"Found {df.shape[0]} commits with bug-related messages.")
df.to_csv("Bug_Mining.csv", index=False)
print("Data saved to Bug_Mining.csv")