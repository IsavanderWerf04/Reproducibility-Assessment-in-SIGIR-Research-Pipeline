"""
github_checker.py
The GitHub Checker node in the LangGraph pipeline.

Sits between the Extractor and Inspector.
Finds GitHub links in C2, C5, and C6 passages and checks if a LICENSE
file exists using the GitHub API. If no LICENSE file is found, fetches
the README as a fallback. Adds results to the passage text so the
Inspector can use them to judge the criteria.
"""

import re
import base64
import requests
from dotenv import load_dotenv
from state import PaperState
import os

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

CRITERIA_TO_CHECK = ["C2", "C5", "C6"]


def extract_github_links(text: str) -> list:
    """Extract all GitHub repository URLs from a text string."""
    pattern = r'https?://github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)'
    matches = re.findall(pattern, text)
    # strip trailing dots or dashes from repo names
    return [(owner, repo.rstrip('.-')) for owner, repo in matches]


def get_headers():
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def check_github_license(owner: str, repo: str) -> dict:
    """Check if a GitHub repository has a LICENSE file using the GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/license"
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        print(f"[GitHub Checker] API status: {response.status_code} for {owner}/{repo}")  # add this lin
        if response.status_code == 200:
            data = response.json()
            license_name = data.get("license", {}).get("name", "Unknown license")
            return {"found": True, "license": license_name}
        elif response.status_code == 404:
            return {"found": False, "license": None}
        else:
            return {"found": None, "error": f"API returned {response.status_code}"}
    except requests.RequestException as e:
        return {"found": None, "error": str(e)}


def get_github_readme(owner: str, repo: str) -> str | None:
    """Fetch and decode the README file from a GitHub repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        if response.status_code == 200:
            data = response.json()
            content = data.get("content", "")
            encoding = data.get("encoding", "")
            if encoding == "base64":
                return base64.b64decode(content).decode("utf-8", errors="ignore")
            return content
        else:
            return None
    except requests.RequestException:
        return None


def enrich_passages_with_github(passages: list, criterion: str) -> list:
    """
    For a list of passages, find GitHub links, check for LICENSE file,
    and fetch README only if no LICENSE file is found.
    Append all results to the passage text.
    """
    updated_passages = []
    for passage in passages:
        text = passage.get("verbatim_text", "")
        links = extract_github_links(text)

        if links:
            github_results = []
            for owner, repo in links:
                print(f"[GitHub Checker] [{criterion}] Checking https://github.com/{owner}/{repo}...")
                repo_info = []

                # check LICENSE file
                license_result = check_github_license(owner, repo)
                if license_result["found"] is True:
                    repo_info.append(f"LICENSE file found: {license_result['license']}")
                    print(f"[GitHub Checker] License found: {license_result['license']}")
                else:
                    if license_result["found"] is False:
                        repo_info.append("No LICENSE file found")
                        print(f"[GitHub Checker] No LICENSE file found, fetching README...")
                    else:
                        repo_info.append(f"Could not verify LICENSE: {license_result.get('error')}")
                        print(f"[GitHub Checker] Could not verify license, fetching README...")

                    # only fetch README if no LICENSE file found
                    readme = get_github_readme(owner, repo)
                    if readme:
                        repo_info.append(f"README content:\n{readme}")
                        print(f"[GitHub Checker] README fetched ({len(readme)} chars).")
                    else:
                        repo_info.append("No README found.")
                        print(f"[GitHub Checker] No README found.")

                github_results.append(
                    f"[GitHub API: https://github.com/{owner}/{repo}]\n" + "\n".join(repo_info)
                )

            passage = dict(passage)
            passage["verbatim_text"] = text + "\n\n" + "\n\n".join(github_results)

        updated_passages.append(passage)
    return updated_passages


def github_checker_node(state: PaperState) -> dict:
    """
    Checks GitHub links found in C2, C5, and C6 passages for license information.
    Fetches README as fallback if no LICENSE file is found.
    Appends results to passage text for the Inspector.
    """
    print("[GitHub Checker] Checking GitHub links in C2, C5, C6 passages...")

    relevant_passages = dict(state["relevant_passages"])
 
    for criterion in CRITERIA_TO_CHECK:
        passages = relevant_passages.get(criterion, [])
        if not passages:
            print(f"[GitHub Checker] No {criterion} passages found, skipping.")
            continue
        relevant_passages[criterion] = enrich_passages_with_github(passages, criterion)

    print(f"[GitHub Checker] Done.")
    return {"relevant_passages": relevant_passages}