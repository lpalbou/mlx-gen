import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Callable

Response = tuple[int, str, dict[str, str]]


class Context7Refresh:
    API = "https://context7.com/api"
    # A longer wait than this is not worth holding a release job for; the run fails visibly instead.
    MAX_RETRY_AFTER_SECONDS = 300

    def __init__(self, key: str, post: Callable[[str, dict], Response] | None = None, sleep=time.sleep):
        self.key = key
        self.post = post or self._post
        self.sleep = sleep

    def run(self, library: str, repository_url: str, label: str) -> int:
        if not self.key:
            print(
                "::warning::the context7 environment has no CONTEXT7_API_KEY credential; "
                f"Context7 was not refreshed for {label}"
            )
            return 0
        status, text, headers = self.post("/v1/refresh", {"libraryName": library})
        if status == 429:
            wait = int(float(headers.get("Retry-After", "0") or 0))
            if 0 < wait <= self.MAX_RETRY_AFTER_SECONDS:
                print(f"Context7 rate limit: waiting {wait}s once as asked, then retrying")
                self.sleep(wait)
                status, text, headers = self.post("/v1/refresh", {"libraryName": library})
        if status == 200:
            print(f"Context7 refresh of {library} started for {label}: {text.strip()}")
            return 0
        if status == 400 and "too-early" in text:
            # Context7 re-indexes a project at most once every 10 days; inside that window it follows its
            # own schedule, so a release is not failed for it.
            print(f"::notice::Context7 did not refresh {library} for {label}: {text.strip()}")
            return 0
        if status == 404 and "library_not_found" in text:
            return self._add(library, repository_url, label)
        print(f"::error::Context7 refresh of {library} for {label} failed: HTTP {status} {text.strip()}")
        return 1

    def _add(self, library: str, repository_url: str, label: str) -> int:
        status, text, _ = self.post("/v2/add/repo/github", {"docsRepoUrl": repository_url})
        if status == 200:
            print(f"Context7 did not know {library}; submitted {repository_url} for {label}: {text.strip()}")
            return 0
        print(f"::error::Context7 could not add {repository_url} for {label}: HTTP {status} {text.strip()}")
        return 1

    def _post(self, path: str, body: dict) -> Response:
        request = urllib.request.Request(
            f"{self.API}{path}",
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, response.read().decode("utf-8", "replace"), dict(response.headers)
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode("utf-8", "replace"), dict(error.headers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask Context7 to re-index this repository's documentation.")
    parser.add_argument("--repository", required=True, help="owner/name, as in GITHUB_REPOSITORY")
    parser.add_argument("--label", required=True, help="What the refresh is for, such as a version or a commit.")
    args = parser.parse_args()
    refresh = Context7Refresh(key=os.environ.get("CONTEXT7_API_KEY", ""))
    sys.exit(
        refresh.run(
            library=f"/{args.repository}",
            repository_url=f"https://github.com/{args.repository}",
            label=args.label,
        )
    )


if __name__ == "__main__":
    main()
