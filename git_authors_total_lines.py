import subprocess
import re
import os
import csv

repos = {
    "activemq": {
        "url": "https://github.com/apache/activemq.git",
        "tags": ["activemq-4.1.2", "activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.8.0"]
    },
    "camel": {
        "url": "https://github.com/apache/camel.git",
        "tags": ["camel-1.3.0", "camel-1.4.0", "camel-2.9.0", "camel-2.10.0", "camel-2.11.0"]
    },
    # "derby": {
    #     "url": "https://github.com/apache/derby.git",
    #     "tags": ["10.2.1.6", "release-10.3.1.4", "release-10.5.1.1"]
    # },
    "groovy": {
        "url": "https://github.com/apache/groovy.git",
        "tags": ["GROOVY_1_5_4", "GROOVY_1_5_7", "GROOVY_1_6_BETA_1", "GROOVY_1_6_BETA_2"]
    },
    "hbase": {
        "url": "https://github.com/apache/hbase.git",
        "tags": ["0.92.2", "0.94.0", "0.95.0", "0.95.2"]
    },
    "hive": {
        "url": "https://github.com/apache/hive.git",
        "tags": ["release-0.7.0", "release-0.9.0", "release-0.10.0", "release-0.12.0"]
    },
    "jruby": {
        "url": "https://github.com/jruby/jruby.git",
        "tags": ["1.3.0", "1.1", "1.4.0", "1.5.0", "1.7.0.preview1"]
    },
    "lucene": {
        "url": "https://github.com/apache/lucene.git",
        "tags": ["releases/lucene/2.2.0", "releases/lucene/2.3.0", "releases/lucene/2.9.0", "releases/lucene/3.0.0", "releases/lucene-solr/3.1"]
    },
    "wicket": {
        "url": "https://github.com/apache/wicket.git",
        "tags": ["wicket-1.2.7", "wicket-1.3.0-incubating-beta-1", "wicket-1.3.0-beta2", "wicket-1.5.3"]
    }
}

def normalize(tag):
    """Normalize tags to allow fuzzy matching."""
    tag = tag.lower()
    tag = tag.replace("_", ".").replace("-", ".")
    tag = re.sub(r"^release\.", "", tag)
    return tag

for project, info in repos.items():
    print(f"\nChecking {project}...")
    try:
        output = subprocess.check_output(
            ["git", "ls-remote", "--tags", info["url"]],
            stderr=subprocess.DEVNULL
        ).decode()
        remote_tags = [line.split("refs/tags/")[1] for line in output.strip().split("\n") if "refs/tags/" in line]

        normalized_remote = {normalize(tag): tag for tag in remote_tags}

        for desired in info["tags"]:
            if normalize(desired) in normalized_remote:
                print(f"Found tag: {normalized_remote[normalize(desired)]}")
            else:
                print(f"Missing tag: {desired}")
    except Exception as e:
        print(f"  Error checking {project}: {e}")

output_dir_diff = "release_commit_data"
os.makedirs(output_dir_diff, exist_ok=True)

for repo_name, repo_info in repos.items():
    repo_url = repo_info["url"]
    repo_tags = repo_info["tags"]
    repo_dir = repo_name

    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

    os.chdir(repo_dir)
    csv_path = os.path.join("..", output_dir_diff, f"{repo_name}_tag_commits_diff_total.csv")
    skipped_csv_path = os.path.join("..", output_dir_diff, f"{repo_name}_tag_commits_skipped_total.csv")

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f, \
         open(skipped_csv_path, mode='w', newline='', encoding='utf-8') as sf:

        writer = csv.writer(f)
        skipped_writer = csv.writer(sf)

        writer.writerow(["Tag", "Commit ID", "File", "Author", "Author Time", "Lines Added", "Lines Deleted"])
        skipped_writer.writerow(["Tag", "Commit ID", "Reason"])

        all_tags = subprocess.check_output(
            ["git", "for-each-ref", "--sort=creatordate", "--format=%(refname:short)", "refs/tags"],
            text=True
        ).strip().splitlines()

        if not all_tags:
            print(f"No tags found in {repo_name}, skipping…")
            os.chdir("..")
            continue

        first_tag = all_tags[0]
        print(f"Earliest tag in {repo_name} is {first_tag}")

        for idx, tag in enumerate(repo_tags):
            if idx == 0:
                continue

            try:
                prev_tag = first_tag

                revlist = subprocess.check_output(
                    ["git", "rev-list", f"{prev_tag}..{tag}"],
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

