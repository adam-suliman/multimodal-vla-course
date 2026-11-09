"""Stage hash-locked CPython 3.11/Linux wheels with resumable HTTP ranges.

Run with Python 3.11 and packaging installed. The output is an optional local
download cache; every artifact is independently verified against the lock files.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

try:
    from packaging.tags import cpython_tags, compatible_tags
    from packaging.utils import parse_wheel_filename
except ImportError:
    from pip._vendor.packaging.tags import cpython_tags, compatible_tags
    from pip._vendor.packaging.utils import parse_wheel_filename


PLATFORMS = [f"manylinux_2_{v}_x86_64" for v in range(39, 16, -1)] + [
    "manylinux2014_x86_64",
    "manylinux2010_x86_64",
    "manylinux1_x86_64",
    "linux_x86_64",
]
TAGS = list(cpython_tags((3, 11), platforms=PLATFORMS)) + list(
    compatible_tags((3, 11), interpreter="cp311", platforms=PLATFORMS)
)
RANK = {tag: n for n, tag in enumerate(TAGS)}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            result.update(block)
    return result.hexdigest()


def metadata(name, version, hashes):
    if name.startswith("isaacsim") or "+cu126" in version or name == "triton":
        base = (
            (
                "https://pypi.nvidia.com/"
                if name.startswith("isaacsim")
                else "https://download.pytorch.org/whl/cu126/"
            )
            + name
            + "/"
        )
        html = urllib.request.urlopen(base, timeout=120).read().decode()
        candidates = []
        for href in re.findall(r'href="([^"]+)"', html):
            url = urllib.parse.urljoin(base, href)
            filename = urllib.parse.unquote(
                urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
            )
            sha = urllib.parse.parse_qs(urllib.parse.urlparse(url).fragment).get(
                "sha256", [None]
            )[0]
            if sha in hashes:
                candidates.append((filename, url.split("#")[0], sha))
    else:
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        data = json.load(urllib.request.urlopen(url, timeout=120))
        candidates = [
            (x["filename"], x["url"], x["digests"]["sha256"])
            for x in data["urls"]
            if x["digests"]["sha256"] in hashes
        ]
    ranked = []
    for filename, url, sha in candidates:
        if filename.endswith(".whl"):
            tags = parse_wheel_filename(filename)[3]
            score = min((RANK[t] for t in tags if t in RANK), default=10**9)
            if score == 10**9:
                continue
        else:
            score = 10**8
        ranked.append((score, filename, url, sha))
    if not ranked:
        raise RuntimeError(
            f"No locked Linux CPython 3.11 distribution: {name}=={version}"
        )
    return sorted(ranked)[0][1:]


def download(directory, filename, url, sha):
    target = directory / filename
    if target.exists() and digest(target) == sha:
        print(f"Verified {filename}", flush=True)
        return
    partial = target.with_suffix(target.suffix + ".partial")
    progress = target.with_suffix(target.suffix + ".ranges.json")
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        ranged = response.status == 206
        size = int(response.headers["Content-Range"].split("/")[-1]) if ranged else None
    if not ranged:
        with urllib.request.urlopen(url, timeout=300) as src, partial.open("wb") as dst:
            while block := src.read(2**20):
                dst.write(block)
    else:
        saved = (
            json.loads(progress.read_text())
            if progress.exists() and partial.exists()
            else {}
        )
        done = set(saved.get("done", [])) if saved.get("sha256") == sha else set()
        chunk = 8 * 2**20
        fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, size)

        def fetch(start):
            end = min(start + chunk, size)
            for attempt in range(6):
                try:
                    req = urllib.request.Request(
                        url, headers={"Range": f"bytes={start}-{end - 1}"}
                    )
                    with urllib.request.urlopen(req, timeout=120) as src:
                        if (
                            src.status != 206
                            or src.headers.get("Content-Range")
                            != f"bytes {start}-{end - 1}/{size}"
                        ):
                            raise RuntimeError("Invalid HTTP range response")
                        data = src.read(end - start + 1)
                    if len(data) != end - start:
                        raise RuntimeError("Truncated download")
                    view = memoryview(data)
                    offset = start
                    while view:
                        count = os.pwrite(fd, view, offset)
                        offset += count
                        view = view[count:]
                    return start
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(attempt + 1)

        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                for start in pool.map(
                    fetch, [s for s in range(0, size, chunk) if s not in done]
                ):
                    done.add(start)
                    os.fsync(fd)
                    progress.write_text(
                        json.dumps({"sha256": sha, "done": sorted(done)})
                    )
        finally:
            os.close(fd)
    if digest(partial) != sha:
        progress.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch: {filename}; retry to replace all ranges")
    partial.replace(target)
    progress.unlink(missing_ok=True)
    print(f"Downloaded and verified {filename}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("locks", nargs="+", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    entries = {}
    for lock in args.locks:
        text = lock.read_text().replace("\\\n", " ")
        for line in text.splitlines():
            match = re.match(r"([\w-]+)==([^\s;]+)(.*)", line)
            if match:
                name, version, rest = match.groups()
                entries[name, version] = set(
                    re.findall(r"--hash=sha256:([a-f0-9]{64})", rest)
                )

    def stage(item):
        (name, version), hashes = item
        for attempt in range(6):
            try:
                print(f"Resolving {name}=={version}", flush=True)
                download(args.directory, *metadata(name, version, hashes))
                return
            except Exception as exc:
                print(f"Retry {name}: {type(exc).__name__}: {exc}", flush=True)
                if attempt == 5:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(stage, entries.items()))


if __name__ == "__main__":
    main()
