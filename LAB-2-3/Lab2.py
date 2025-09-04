#!/usr/bin/env python3

import subprocess
import csv
import os
import time
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check if transformers is installed
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    transformers_available = True
except ImportError:
    transformers_available = False
    print("Warning: transformers library not found. Will use rule-based analysis instead.")
    print("To install: pip install transformers torch")

# Check if groq is installed for Grok access
try:
    import groq
    groq_available = True
except ImportError:
    groq_available = False
    print("Warning: groq library not found. Will use transformers for commit messages.")
    print("To install: pip install groq")

# Constants
INPUT_CSV = 'fixes.csv'
OUTPUT_CSV = 'commit_analysis.csv'
# Number of concurrent git operations
MAX_WORKERS = 4

def decode_output(binary_output):
    """Safely decode command output with multiple encodings."""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            return binary_output.decode(encoding)
        except UnicodeDecodeError:
            continue
    # If all decodings fail, use 'replace' mode with utf-8
    return binary_output.decode('utf-8', errors='replace')

def get_file_content_at_commit(commit_hash, file_path):
    """Get file content at specific commit."""
    try:
        cmd = ['git', 'show', f'{commit_hash}:{file_path}']
        result = subprocess.run(cmd, capture_output=True)
        return decode_output(result.stdout)
    except subprocess.CalledProcessError:
        return ""

def get_file_diff(commit_hash, file_path):
    """Get the diff for a specific file at a commit."""
    cmd = ['git', 'show', '--format=', '--patch', f'{commit_hash}', '--', file_path]
    result = subprocess.run(cmd, capture_output=True)
    return decode_output(result.stdout)

def get_previous_commit(commit_hash):
    """Get the parent commit hash."""
    try:
        cmd = ['git', 'rev-parse', f'{commit_hash}^']
        result = subprocess.run(cmd, capture_output=True)
        return decode_output(result.stdout).strip()
    except subprocess.CalledProcessError:
        return ""

def get_commit_files_data(commit_hash, files):
    """Get content and diff for all files in a commit in parallel."""
    # Filter out empty file paths once
    valid_files = [f for f in files if f]
    if not valid_files:
        return []
    
    previous_commit = get_previous_commit(commit_hash)
    
    # Use ThreadPoolExecutor to parallelize git operations
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create all futures at once
        futures = {}
        
        for file_path in valid_files:
            # Submit current content
            futures[executor.submit(get_file_content_at_commit, commit_hash, file_path)] = ('current', file_path)
            
            # Submit previous content if available
            if previous_commit:
                futures[executor.submit(get_file_content_at_commit, previous_commit, file_path)] = ('previous', file_path)
            
            # Submit diff
            futures[executor.submit(get_file_diff, commit_hash, file_path)] = ('diff', file_path)
        
        # Collect all results in a single pass
        file_contents = {}
        prev_file_contents = {}
        file_diffs = {}
        
        for future in as_completed(futures):
            result_type, file_path = futures[future]
            try:
                result = future.result()
                if result_type == 'current':
                    file_contents[file_path] = result
                elif result_type == 'previous':
                    prev_file_contents[file_path] = result
                elif result_type == 'diff':
                    file_diffs[file_path] = result
            except Exception as e:
                print(f"Error getting {result_type} for {file_path}: {e}")
                # Set default empty values
                if result_type == 'current':
                    file_contents[file_path] = ""
                elif result_type == 'previous':
                    prev_file_contents[file_path] = ""
                elif result_type == 'diff':
                    file_diffs[file_path] = ""
    
    # Build result list efficiently
    return [
        {
            'file_path': file_path,
            'current_content': file_contents.get(file_path, ""),
            'previous_content': prev_file_contents.get(file_path, ""),
            'diff': file_diffs.get(file_path, "")
        }
        for file_path in valid_files
    ]

def analyze_diff(diff):
    """Analyze the diff to determine the fix type."""
    try:
        # Load model and tokenizer (only load once)
        if not hasattr(analyze_diff, 'tokenizer') or not hasattr(analyze_diff, 'model'):
            if not transformers_available:
                print("Transformers not available, using fallback analysis")
                return "Code Change"
            
            print("Loading CommitPredictorT5 model (this will be done only once)...")
            analyze_diff.tokenizer = AutoTokenizer.from_pretrained("mamiksik/CommitPredictorT5")
            analyze_diff.model = AutoModelForSeq2SeqLM.from_pretrained("mamiksik/CommitPredictorT5")
            print("Model loaded successfully.")
        
        # Truncate diff if too large
        truncated_diff = diff[:4000] if len(diff) > 4000 else diff
        
        # Prepare prompt specifically for fix type classification
        fix_type_prompt = f"tell commmit message for this code change.\n\n{truncated_diff}"
        
        # Generate fix type (short and focused)
        inputs_fix = analyze_diff.tokenizer(fix_type_prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs_fix = analyze_diff.model.generate(
            inputs_fix.input_ids, 
            max_length=15,  # Very short to get just the category
            num_beams=5,    # Use beam search for more focused results
            do_sample=True,
            top_p=0.9,
            early_stopping=True
        )
        
        # Decode output
        fix_type = analyze_diff.tokenizer.decode(outputs_fix[0], skip_special_tokens=True)
        
        return fix_type
        
    except Exception as e:
        print(f"Error during LLM inference: {str(e)}")
        # Fallback to simple analysis
        return "Code Change"

def generate_commit_message_with_groq(diff):
    """Generate a commit message using Groq API."""
    if not groq_available:
        return "Updated code with necessary changes"
    
    try:
        # Initialize Groq client with API key
        GROQ_API_KEY = "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"
        client = groq.Groq(api_key=GROQ_API_KEY)
        
        # Truncate diff if too large
        truncated_diff = diff[:4000] if len(diff) > 4000 else diff
        
        # Create prompt for commit message generation
        prompt = f"""You Just output the commit message for given code diff.
        
        Code changes:
        {truncated_diff}
        
        Commit message:"""
        
        # Generate response using Groq API
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # Using Llama3 model through Groq (fast and free tier)
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=15,
            temperature=0.3,  # Lower temperature for more consistent results
            top_p=0.9
        )
        
        # Extract the generated message
        commit_message = response.choices[0].message.content.strip()
        
        # Clean up the message (remove quotes if present)
        if commit_message.startswith('"') and commit_message.endswith('"'):
            commit_message = commit_message[1:-1]
        
        return commit_message
        
    except Exception as e:
        print(f"Error generating commit message with Groq: {str(e)}")
        return "Updated code with necessary changes"

def process_all_commits():
    """Process all commits in the input CSV and create a comprehensive analysis CSV."""
    # Define CSV headers
    headers = [
        'Hash', 
        'Message', 
        'Filename', 
        'Source Code (before)', 
        'Source Code (current)', 
        'Diff', 
        'LLM Inference (fix type)', 
        'Rectified Message'
    ]
    
    try:
        # Read the input CSV
        with open(INPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            commits = list(reader)
        
        print(f"Found {len(commits)} commits to process")
    except Exception as e:
        print(f"Error reading {INPUT_CSV}: {str(e)}")
        return
    
    # Create or overwrite the output CSV with headers
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
    
    # Process each commit
    for i, commit_entry in enumerate(commits):
        try:
            commit_hash = commit_entry['Hash'].strip()
            commit_message = commit_entry['Message'].strip()
            
            print(f"\nProcessing commit {i+1}/{len(commits)}: {commit_hash}")
            print(f"Message: {commit_message}")
            
            # Parse the list of modified files
            if 'List of modified files' in commit_entry:
                try:
                    files = eval(commit_entry['List of modified files'])
                except (SyntaxError, NameError):
                    # Handle case where the list might be a string format
                    files_str = commit_entry['List of modified files'].strip("[]'\"")
                    files = [f.strip("'\" ") for f in files_str.split(",")]
            else:
                print(f"No files found for commit {commit_hash}")
                continue
            
            # Skip if no files to process
            if not files or (len(files) == 1 and not files[0]):
                print(f"No files to process for commit {commit_hash}")
                continue
            
            # Get all file data and combined diff in one operation
            file_data = get_commit_files_data(commit_hash, files)
            
            # Analyze the entire commit with one LLM call
            # print(f"Analyzing commit with LLM...")
            # fix_type, rectified_message = analyze_diff(combined_diff)
            # print(f"Analysis complete: {fix_type}")
            
            # Write each file's data to the CSV
            for file_info in file_data:
                file_path = file_info['file_path']
                current_content = file_info['current_content']
                previous_content = file_info['previous_content']
                file_diff = file_info['diff']
                # Use transformers for fix type classification
                fix_type = analyze_diff(file_diff)
                # Use Groq for commit message generation
                rectified_message = generate_commit_message_with_groq(file_diff)
                
                # Prepare row for CSV
                row = {
                    'Hash': commit_hash,
                    'Message': commit_message,
                    'Filename': file_path,
                    'Source Code (before)': previous_content,
                    'Source Code (current)': current_content,
                    'Diff': file_diff,
                    'LLM Inference (fix type)': fix_type,
                    'Rectified Message': rectified_message
                }
                print(row)
                # Append to the CSV
                with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=headers)
                    writer.writerow(row)
                
                print(f"  Added analysis for {file_path}")
            
        except Exception as e:
            print(f"Error processing commit {commit_hash}: {str(e)}")
            continue
    
    print(f"\nAnalysis completed. Results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    print("Processing all commits...")
    process_all_commits()
