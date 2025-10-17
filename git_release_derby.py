import subprocess
import re
import os
import csv

repos = {
    "derby": {
        "url": "https://github.com/apache/derby.git",
        "tags": ["10.1.2.1", "10.2.1.6",
                 "10.3.1.4", "10.5.1.1"
                 ]
    },
}

def normalize(tag):
    """Normalize tags to allow fuzzy matching."""
    tag = tag.lower()
    tag = tag.replace("_", ".").replace("-", ".")
    tag = re.sub(r"^release\.", "", tag)
    return tag

output_dir_diff = "release_commit_data"
os.makedirs(output_dir_diff, exist_ok=True)
tag_date = {"10.1.2.1": "2005-12-06 18:02:39 +0000",
            "10.2.1.6": "2006-10-06 21:24:22 +0000",
            "10.3.1.4": "2007-08-11 02:11:00 +0000",
            "10.5.1.1": "2009-05-01 05:11:46 +0000",
            }
for repo_name, repo_info in repos.items():
    repo_url = repo_info["url"]
    repo_tags = repo_info["tags"]
    repo_dir = repo_name

    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

    os.chdir(repo_dir)
    csv_path = os.path.join("..", output_dir_diff, f"{repo_name}_tag_commits_diff.csv")
    skipped_csv_path = os.path.join("..", output_dir_diff, f"{repo_name}_tag_commits_skipped.csv")

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f, \
         open(skipped_csv_path, mode='w', newline='', encoding='utf-8') as sf:

        writer = csv.writer(f)
        skipped_writer = csv.writer(sf)

        writer.writerow(["Tag", "Commit ID", "File", "Author", "Author Time", "Lines Added", "Lines Deleted"])
        skipped_writer.writerow(["Tag", "Commit ID", "Reason"])

        for idx, tag in enumerate(repo_tags):
            if idx == 0:
                continue

            try:
                prev_tag = repo_tags[idx - 1]

                revlist = subprocess.check_output(
                    ["git", "rev-list", f'--since="{tag_date[prev_tag]}"', f'--until="{tag_date[tag]}"', "HEAD"],
                    text=True
                ).strip().split('\n')

                print(f"\n{repo_name} | {prev_tag}..{tag}")
                print(f"Total commits from rev-list: {len(revlist)}")

                for commit in revlist:
                    commit = commit.strip()
                    if not commit:
                        skipped_writer.writerow([tag, commit, "Empty commit ID"])
                        continue

                    try:
                        meta = subprocess.check_output(
                            ["git", "show", "--no-patch", "--pretty=format:%H%n%an <%ae>%n%ad", commit],
                            text=True
                        ).strip().split('\n')
                    except subprocess.CalledProcessError as e:
                        skipped_writer.writerow([tag, commit, f"Error getting meta: {e}"])
                        continue

                    if len(meta) < 3:
                        skipped_writer.writerow([tag, commit, "Meta too short"])
                        continue

                    commit_id, author, author_time = meta[:3]

                    try:
                        files = subprocess.check_output(
                            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_id],
                            text=True
                        ).strip().split('\n')
                    except subprocess.CalledProcessError as e:
                        skipped_writer.writerow([tag, commit_id, f"Error getting files: {e}"])
                        continue

                    files_clean = [f.strip() for f in files if f.strip()]
                    if not files_clean:
                        skipped_writer.writerow([tag, commit_id, "No files changed"])
                        continue

                    try:
                        numstat_output = subprocess.check_output(
                            ["git", "diff-tree", "--numstat", "-r", commit_id],
                            text=True
                        ).strip().split('\n')
                        file_line_changes = {}
                        for line in numstat_output:
                            parts = line.split('\t')
                            if len(parts) >= 3:
                                added, deleted, fname = parts
                                try:
                                    added = int(added) if added != '-' else 0
                                    deleted = int(deleted) if deleted != '-' else 0
                                except ValueError:
                                    added, deleted = 0, 0
                                file_line_changes[fname] = (added, deleted)
                    except subprocess.CalledProcessError as e:
                        skipped_writer.writerow([tag, commit_id, f"Error getting line stats: {e}"])
                        continue

                    for file in files_clean:
                        added, deleted = file_line_changes.get(file, (0, 0))
                        writer.writerow([tag, commit_id, file, author, author_time, added, deleted])

            except subprocess.CalledProcessError as e:
                print(f"Error processing tag {tag} in {repo_name}: {e}")

    os.chdir("..")
    print(f"Saved diff commits for {repo_name} in {output_dir_diff}")
    print(f"Saved skipped commits for {repo_name} in {output_dir_diff}")