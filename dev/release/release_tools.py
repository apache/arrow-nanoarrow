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

"""
Python implementations of various release tasks

Use `python release_tools.py --help` for usage
"""

import argparse
import re
import subprocess
import os


def git(*args):
    out = subprocess.run(["git"] + list(args), stdout=subprocess.PIPE)
    return out.stdout.decode().strip().splitlines()


def src_path(*args):
    release_dir = os.path.dirname(__file__)
    relative_path = os.path.join(release_dir, "..", "..", *args)
    return os.path.abspath(relative_path)


def file_regex_replace(pattern, replacement, path):
    with open(path) as f:
        content = f.read()

    # It is usually good to know if zero items are about to be replaced
    if re.search(pattern, content) is None:
        raise ValueError(f"file {path} does not contain pattern '{pattern}'")

    content = re.sub(pattern, replacement, content)
    with open(path, "w") as f:
        f.write(content)


def find_last_dev_tag():
    """Finds the commit of the last version bump

    Note that this excludes changes that happened during the release
    process but were not picked into the release branch.
    """
    last_dev_tag = git(
        "describe", "--match", "apache-arrow-nanoarrow-*.dev", "--tags", "--abbrev=0"
    )[0]
    last_version = re.search(r"[0-9]+\.[0-9]+\.[0-9]+", last_dev_tag).group(0)
    sha = git("rev-list", "-n", "1", last_dev_tag)[0]
    return last_version, sha


def find_commits_since(begin_sha, end_sha="HEAD"):
    lines = git("log", "--pretty=oneline", f"{begin_sha}..{end_sha}")
    return lines


def add_set_python_dev_version_subparser(subparsers):
    subparsers.add_parser(
        "set_python_dev_version",
        description=(
            "Set the Python package development version based on "
            "the number of commits since the last version bump"
        ),
    )


def set_python_dev_version_command(args):
    _, last_dev_tag = find_last_dev_tag()
    dev_distance = len(find_commits_since(last_dev_tag))

    version_file = src_path("python", "src", "nanoarrow", "_static_version.py")
    file_regex_replace(
        r'"([0-9]+\.[0-9]+\.[0-9]+)\.dev[0-9]+"',
        f'"\\1.dev{dev_distance}"',
        version_file,
    )


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
    _, sha = find_last_dev_tag()
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


def add_changelog_parser(subparsers):
    parser = subparsers.add_parser(
        "changelog", description="Generate and/or append new CHANGELOG.md content"
    )
    parser.add_argument(
        "unreleased_version",
        nargs="?",
        help=(
            "Prepend heading text ## nanoarrow [unreleased_version]) "
            "to the latest entries"
        ),
    )
    parser.add_argument(
        "changelog_file",
        nargs="?",
        help="If specified, append new changelog content to this file",
    )


def changelog_command(args):
    print(render_new_changelog(args.unreleased_version, args.changelog_file))


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(
        description="Python functions automating various pieces of release tasks",
    )

    subparsers = parser.add_subparsers(
        title="subcommands", dest="subcommand", required=True
    )
    add_changelog_parser(subparsers)
    add_set_python_dev_version_subparser(subparsers)

    args = parser.parse_args(sys.argv[1:])
    if args.subcommand == "changelog":
        changelog_command(args)
    elif args.subcommand == "set_python_dev_version":
        set_python_dev_version_command(args)
