# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import re
import subprocess

"""
A Python script to update CHANGELOG.md

This is similar to cz changelog except is specific to the nanoarrow/Apache
release/tag format. The usage is:

mv CHANGELOG.md CHANGELOG.md.bak
python changelog.py <new version> CHANGELOG.md.bak > CHANGELOG.md
rm CHANGELOG.md.bak

This can be run more than once (e.g., for multiple release candidates) and will
overwrite the changelog section for <new version>. It always has one newline
at the end and does not mangle changelog sections for previous versions. It
groups commit types (e.g., feat, fix, refactor) and groups top-level components.
"""


def git(*args):
    out = subprocess.run(["git"] + list(args), stdout=subprocess.PIPE)
    return out.stdout.decode("UTF-8").splitlines()


def find_last_release_sha():
    """Finds the commit of the last release

    For the purposes of the changelog, this is the commit where the versions
    were bumped. This would exclude changes that happened during the release
    process but were not picked into the release branch.
    """
    last_dev_tag = git(
        "describe", "--match", "apache-arrow-nanoarrow-*.dev", "--tags", "--abbrev=0"
    )[0]
    return git("rev-list", "-n", "1", last_dev_tag)[0]


def find_commits_since(begin_sha, end_sha="HEAD"):
    lines = git("log", "--pretty=oneline", f"{begin_sha}..{end_sha}")
    return lines


def parse_commits(lines):
    commit_pattern = (
        r"^(?P<sha>[a-z0-9]{40}) (?P<type>[a-z]+)"
        r"(\((?P<component>[a-zA-Z0-9/_-]+)\))?:\s*"
        r"(?P<message>.*)$"
    )

    out = []
    for line in lines:
        parsed = re.search(commit_pattern, line)
        if parsed:
            out.append(parsed.groupdict())

    return out


def group_commits_by_type(parsed):
    grouped = {}

    for item in parsed:
        if item["type"] not in grouped:
            grouped[item["type"]] = []

        grouped[item["type"]].append(item)

    return grouped


def group_commits_by_top_level_component(parsed):
    grouped = {}

    for item in parsed:
        component = item["component"]
        top_level_component = component.split("/")[0] if component else ""
        if top_level_component not in grouped:
            grouped[top_level_component] = []

        grouped[top_level_component].append(item)

    return grouped


def render_version_content(parsed):
    grouped = group_commits_by_type(parsed)
    for category in grouped:
        grouped[category] = group_commits_by_top_level_component(grouped[category])

    out_lines = []
    for category in sorted(grouped):
        if category in ("chore", "ci"):
            continue

        out_lines.append(f"### {category.capitalize()}")
        out_lines.append("")

        for component in sorted(grouped[category]):
            for item in grouped[category][component]:
                component = item["component"]
                prefix = f"**{component}**: " if component else ""
                message = item["message"]
                out_lines.append(f"- {prefix}{message}")

        out_lines.append("")

    if out_lines[-1] == "":
        out_lines.pop(-1)
    return "\n".join(out_lines)


def parse_changelog(content):
    header, content = re.split(r"# nanoarrow Changelog", content)
    header += "# nanoarrow Changelog"
    content = content.strip()

    version_split = re.split(r"(^|\n)##\s+nanoarrow ([^\n]*)", content)
    version_split.pop(0)

    version_content = {}
    for i in range(0, len(version_split), 3):
        version_content[version_split[i + 1]] = version_split[i + 2].strip()

    return header, version_content


def render_new_changelog(unreleased_version=None, changelog_file=None):
    sha = find_last_release_sha()
    commits = find_commits_since(sha)
    parsed = parse_commits(commits)

    latest_version_content = render_version_content(parsed)

    if changelog_file is None and unreleased_version is None:
        return latest_version_content

    if changelog_file is None:
        return f"## nanoarrow {unreleased_version}\n\n" + latest_version_content

    with open(changelog_file) as f:
        changelog_content = f.read()

    header, version_content = parse_changelog(changelog_content)

    version_content[unreleased_version] = latest_version_content

    out_lines = []
    out_lines.append(header)
    out_lines.append("")

    for version, content in version_content.items():
        out_lines.append(f"## nanoarrow {version}")
        out_lines.append("")
        out_lines.append(content)
        out_lines.append("")

    if out_lines[-1] == "":
        out_lines.pop(-1)
    return "\n".join(out_lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        changelog_file = sys.argv[2]
        unreleased_version = sys.argv[1]
    elif len(sys.argv) >= 2:
        changelog_file = None
        unreleased_version = sys.argv[1]
    else:
        changelog_file = None
        unreleased_version = None

    print(render_new_changelog(unreleased_version, changelog_file))
