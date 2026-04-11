"""Kaggle automation client using the new KGAT Bearer token.

Uses the REST API directly (not the legacy kaggle CLI) so we can:
  - upload a dataset
  - push a notebook kernel
  - check kernel status
  - download kernel outputs

Token comes from KAGGLE_API_TOKEN in .env (format: KGAT_xxx).
All state is kept local; no ~/.kaggle/kaggle.json needed.
"""
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

BASE = "https://www.kaggle.com/api/v1"


def load_token():
    """Load KGAT token from .env. Returns the token string."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("KAGGLE_API_TOKEN="):
                return line.split("=", 1)[1].strip()
    tok = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KGAT")
    if not tok:
        raise RuntimeError("KAGGLE_API_TOKEN not found in .env or env vars")
    return tok


class Kaggle:
    def __init__(self, token=None):
        self.token = token or load_token()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
        })
        self._username = None

    # ---------- Account ----------
    def whoami(self):
        r = self.session.get(f"{BASE}/hello", timeout=15)
        r.raise_for_status()
        data = r.json()
        self._username = data.get("userName")
        return self._username

    @property
    def username(self):
        return self._username or self.whoami()

    # ---------- Datasets ----------
    def dataset_exists(self, owner, slug):
        """Check if a dataset already exists."""
        r = self.session.get(
            f"{BASE}/datasets/view/{owner}/{slug}",
            timeout=15, allow_redirects=False,
        )
        return r.status_code == 200

    def _upload_file(self, file_path):
        """Upload one file to Kaggle's blob store. Returns the blob token.

        Kaggle's dataset upload uses a 2-step flow:
          1) POST /datasets/upload/file/{size}/{lastModEpoch}  body={fileName}
             -> returns {"token": ..., "createUrl": ...}  (GCS resumable URL)
          2) PUT to createUrl with the raw file bytes
        """
        fp = Path(file_path)
        size = fp.stat().st_size
        last_mod = int(fp.stat().st_mtime)

        r = self.session.post(
            f"{BASE}/datasets/upload/file/{size}/{last_mod}",
            json={"fileName": fp.name},
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Upload init failed ({r.status_code}): {r.text[:400]}")
        info = r.json()
        blob_token = info.get("token")
        create_url = info.get("createUrl")
        if not blob_token or not create_url:
            raise RuntimeError(f"Missing token/createUrl in response: {info}")

        # 2) Stream the file to the presigned GCS URL (no auth header needed —
        #    the URL itself is signed)
        with open(fp, "rb") as f:
            put = requests.put(
                create_url, data=f,
                headers={"Content-Type": "application/octet-stream"},
                timeout=3600,  # long timeout for large files
            )
        if put.status_code not in (200, 201):
            raise RuntimeError(f"Upload PUT failed ({put.status_code}): {put.text[:400]}")
        return blob_token

    def create_dataset(self, title, slug, folder, description="", public=False,
                       license_name="CC0-1.0"):
        """Create a new dataset by uploading every file in folder.

        Args:
          title: human-readable dataset title
          slug:  url-safe slug (e.g., "animtoon-iconscout-v4")
          folder: path to a directory whose files will be uploaded
          public: if True, dataset is public
        """
        owner = self.username
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(folder)

        files = [f for f in sorted(folder.iterdir()) if f.is_file()]
        if not files:
            raise RuntimeError(f"No files in {folder}")

        print(f"  Uploading {len(files)} file(s) to blob store...")
        blob_tokens = []
        for f in files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"    {f.name} ({size_mb:.1f} MB)")
            tok = self._upload_file(f)
            blob_tokens.append({"token": tok, "description": f.name})

        # 3) Create the dataset record referencing the uploaded blobs
        body = {
            "title": title,
            "slug": slug,
            "ownerSlug": owner,
            "licenseName": license_name,
            "subtitle": description[:80] if description else title,
            "description": description,
            "isPrivate": not public,
            "files": blob_tokens,
        }
        r = self.session.post(f"{BASE}/datasets/create/new", json=body, timeout=60)
        if r.status_code != 200:
            print(f"  ERROR {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
        data = r.json()
        if data.get("status") == "Error" or data.get("statusNullable") == "Error":
            err = data.get("error") or data.get("errorNullable")
            raise RuntimeError(f"Kaggle API error: {err}")
        url = data.get("url") or data.get("urlNullable")
        print(f"  Dataset created: {url}")
        return f"{owner}/{slug}"

    def create_or_version_dataset(self, title, slug, folder, description="",
                                  public=False):
        """Create a new dataset or push a new version if it exists."""
        owner = self.username
        folder = Path(folder)
        files = [f for f in sorted(folder.iterdir()) if f.is_file()]
        if not files:
            raise RuntimeError(f"No files in {folder}")

        if self.dataset_exists(owner, slug):
            print(f"  Dataset {owner}/{slug} exists — pushing new version")
            return self._version_dataset(owner, slug, files)
        else:
            print(f"  Creating new dataset {owner}/{slug}")
            return self.create_dataset(title, slug, folder, description, public)

    def _version_dataset(self, owner, slug, files):
        print(f"  Uploading {len(files)} file(s)...")
        blob_tokens = []
        for f in files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"    {f.name} ({size_mb:.1f} MB)")
            tok = self._upload_file(f)
            blob_tokens.append({"token": tok, "description": f.name})
        body = {
            "versionNotes": f"Auto-update {time.strftime('%Y-%m-%d %H:%M')}",
            "files": blob_tokens,
        }
        r = self.session.post(
            f"{BASE}/datasets/create/version/{owner}/{slug}",
            json=body, timeout=60,
        )
        if r.status_code != 200:
            print(f"  ERROR {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
        return f"{owner}/{slug}"

    # ---------- Kernels (notebooks) ----------
    def push_kernel(self, notebook_path, slug, dataset_refs=None,
                    enable_gpu=True, enable_internet=True, private=True,
                    title=None, kernel_type="notebook"):
        """Push a Jupyter notebook to run on Kaggle.

        Args:
          notebook_path: path to .ipynb file
          slug: kernel slug under your account
          dataset_refs: list of 'owner/slug' dataset refs to attach
          enable_gpu: T4 GPU
          enable_internet: allows pip install + HF downloads
        """
        nb_path = Path(notebook_path)
        with open(nb_path, encoding="utf-8") as f:
            notebook_json = f.read()

        owner = self.username
        body = {
            "slug": slug,
            "newTitle": title or slug,
            "text": notebook_json,
            "language": "python",
            "kernelType": kernel_type,
            "isPrivate": private,
            "enableGpu": enable_gpu,
            "enableInternet": enable_internet,
            "datasetDataSources": dataset_refs or [],
            "competitionDataSources": [],
            "kernelDataSources": [],
            "modelDataSources": [],
            "categoryIds": [],
        }
        r = self.session.post(f"{BASE}/kernels/push", json=body, timeout=60)
        if r.status_code != 200:
            print(f"  ERROR {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
        data = r.json()
        url = data.get("url") or data.get("urlNullable", "")
        ref = data.get("ref") or data.get("refNullable", f"{owner}/{slug}")
        version = data.get("versionNumber") or data.get("versionNumberNullable")
        print(f"  Kernel pushed: {url}  (version {version})")
        return ref

    def kernel_status(self, ref):
        """Get run status for a kernel. Returns dict with 'status' and 'failureMessage'."""
        owner, slug = ref.split("/", 1)
        r = self.session.get(
            f"{BASE}/kernels/status/{owner}/{slug}",
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def wait_for_kernel(self, ref, poll_seconds=30, max_hours=10):
        """Poll until the kernel finishes. Returns final status."""
        start = time.time()
        last_status = None
        while True:
            try:
                info = self.kernel_status(ref)
                status = info.get("status") or info.get("statusNullable", "unknown")
            except Exception as e:
                print(f"  (status error: {e})")
                time.sleep(poll_seconds)
                continue
            if status != last_status:
                elapsed_min = (time.time() - start) / 60
                print(f"  [{elapsed_min:5.1f} min] status: {status}")
                last_status = status
            if status.lower() in ("complete", "error", "cancelled", "cancelerequested"):
                return info
            if (time.time() - start) / 3600 > max_hours:
                print(f"  Timeout after {max_hours}h")
                return info
            time.sleep(poll_seconds)

    def download_kernel_output(self, ref, dest_dir):
        """Download all output files from a kernel run."""
        owner, slug = ref.split("/", 1)
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        r = self.session.get(
            f"{BASE}/kernels/output?userName={owner}&kernelSlug={slug}",
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        files = data.get("files") or data.get("filesNullable") or []
        for f in files:
            url = f.get("url")
            name = f.get("fileName") or Path(url).name
            print(f"  downloading {name}")
            resp = self.session.get(url, timeout=600)
            resp.raise_for_status()
            (dest / name).write_bytes(resp.content)
        return dest


if __name__ == "__main__":
    # Smoke test
    k = Kaggle()
    print(f"Authenticated as: {k.whoami()}")
