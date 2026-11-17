"""Download and extract the UCF101 dataset (only the configured classes are extracted).

Usage:
    python -m data.download                 # download archive + splits, extract selected classes
    python -m data.download --skip-download # only extract an archive you placed manually

Manual fallback (if the download fails):
    1. Download https://www.crcv.ucf.edu/data/UCF101/UCF101.rar  (~6.5 GB)
    2. Download https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
    3. Place both files in  motionsense/datasets/raw/
    4. Run:  python -m data.download --skip-download
"""
from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import CLASSES, RAW_DIR, SPLITS_DIR, SPLITS_URL, UCF101_URL, UCF_DIR  # noqa: E402

MANUAL = __doc__.split("Manual fallback")[1]


def download(url: str, dest: Path, retries: int = 30) -> None:
    """Resumable streaming download with retries (HTTP Range)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            head = requests.head(url, timeout=30, allow_redirects=True)
            total = int(head.headers.get("Content-Length", 0))
            have = part.stat().st_size if part.exists() else 0
            if total and have >= total:
                break
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                if mode == "wb":
                    have = 0
                t0, last = time.time(), 0.0
                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - last > 5:
                            last = time.time()
                            pct = 100 * have / total if total else 0
                            mbps = have / 1e6 / max(time.time() - t0, 1e-3)
                            print(f"  {dest.name}: {have/1e9:.2f}/{total/1e9:.2f} GB ({pct:.1f}%)", flush=True)
            if not total or part.stat().st_size >= total:
                break
        except (requests.RequestException, OSError) as e:
            print(f"  attempt {attempt} failed: {e}; retrying in 5 s", flush=True)
            time.sleep(5)
    else:
        raise RuntimeError(f"Download failed after {retries} attempts.\n{MANUAL}")
    part.replace(dest)
    print(f"  saved {dest} ({dest.stat().st_size/1e9:.2f} GB)")


# --------------------------------------------------------------------------------------
# Selective download: fetch only the byte ranges of the selected classes.
# UCF101.rar is a non-solid RAR4 archive whose entries are ordered by class directory, so
# each class occupies one contiguous byte range. We binary-search the remote archive with
# HTTP Range requests to locate those ranges, download them, and rebuild a reduced RAR
# (marker + main header + selected file blocks + end block). UnRAR verifies every file CRC.
# --------------------------------------------------------------------------------------
RAR_MARKER = b"Rar!\x1a\x07\x00"
RAR_END = bytes.fromhex("c43d7b00400700")


class RemoteRar:
    def __init__(self, url: str):
        self.url = url
        self.s = requests.Session()
        self.size = int(self.s.head(url, timeout=30, allow_redirects=True).headers["Content-Length"])

    def get(self, start: int, length: int) -> bytes:
        end = min(start + length, self.size) - 1
        for attempt in range(8):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
                if r.status_code != 206:
                    raise RuntimeError(f"server ignored Range request (HTTP {r.status_code})")
                return r.content
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("range request failed repeatedly")

    @staticmethod
    def parse_header(buf: bytes, p: int):
        """Parse a RAR4 file header starting at buf[p]; returns dict or None (CRC-validated)."""
        import zlib
        if p + 32 > len(buf) or buf[p + 2] != 0x74:
            return None
        crc, flags, hsize = struct.unpack_from("<HxHH", buf, p)
        if not 32 <= hsize <= 1024 or p + hsize > len(buf):
            return None
        if zlib.crc32(buf[p + 2:p + hsize]) & 0xFFFF != crc:
            return None
        pack = struct.unpack_from("<I", buf, p + 7)[0]
        nlen = struct.unpack_from("<H", buf, p + 26)[0]
        off = p + 32
        if flags & 0x100:  # large file: high 32 bits of sizes
            pack |= struct.unpack_from("<I", buf, p + 32)[0] << 32
            off += 8
        name = buf[off:off + nlen].split(b"\x00")[0].decode("latin-1").replace("\\", "/")
        if not name.startswith("UCF-101/") or name.count("/") < 2:
            return None
        return {"hsize": hsize, "pack": pack, "class": name.split("/")[1], "name": name}

    def first_header_at_or_after(self, offset: int):
        chunk = 1 << 20
        while chunk <= 32 << 20 and offset < self.size:
            buf = self.get(offset, chunk)
            p = buf.find(b"\x74")
            while p != -1:
                h = self.parse_header(buf, p - 2) if p >= 2 else None
                if h:
                    h["offset"] = offset + p - 2
                    h["end"] = h["offset"] + h["hsize"] + h["pack"]
                    return h
                p = buf.find(b"\x74", p + 1)
            chunk *= 2
        return None

    def header_at(self, offset: int):
        h = self.parse_header(self.get(offset, 1024), 0)
        if h:
            h["offset"], h["end"] = offset, offset + h["hsize"] + h["pack"]
        return h

    def class_start(self, key, target: str, lo: int) -> int:
        """Byte offset of the first entry whose class sorts >= target (archive order = sorted)."""
        hi, t = self.size, key(target)
        while hi - lo > (4 << 20):
            mid = (lo + hi) // 2
            h = self.first_header_at_or_after(mid)
            if h is None or key(h["class"]) >= t:
                hi = mid
            else:
                lo = h["end"]
        h = self.header_at(lo) or self.first_header_at_or_after(lo)
        while h is not None and key(h["class"]) < t:
            h = self.header_at(h["end"]) if h["end"] < self.size - 7 else None
        return h["offset"] if h else self.size - 7


def selective_download(classes: list[str], out_rar: Path, workers: int = 6) -> None:
    from concurrent.futures import ThreadPoolExecutor
    remote = RemoteRar(UCF101_URL)
    print(f"Remote archive: {remote.size/1e9:.2f} GB — locating class ranges ...")
    head = remote.get(0, 512)
    assert head.startswith(RAR_MARKER), "unexpected archive format (not RAR4)"
    main_hsize = struct.unpack_from("<H", head, 12)[0]
    if struct.unpack_from("<H", head, 10)[0] & 0x0008:
        raise RuntimeError("archive is solid; selective download impossible — use full download")
    preamble = head[:7 + main_hsize]
    first = 7 + main_hsize

    split_idx = SPLITS_DIR / "classInd.txt"
    all_classes = [l.split()[1] for l in split_idx.read_text().splitlines() if l.strip()]
    order = sorted(all_classes, key=str.lower)
    rank = {c: i for i, c in enumerate(order)}
    key = lambda c: rank.get(c, -1)  # noqa: E731

    ranges = {}
    for c in sorted(classes, key=key):
        start = remote.class_start(key, c, first)
        nxt = order[rank[c] + 1] if rank[c] + 1 < len(order) else None
        end = remote.class_start(key, nxt, start) if nxt else remote.size - 7
        h = remote.header_at(start)
        if h is None or h["class"] != c:
            raise RuntimeError(f"could not locate class {c} in the archive")
        ranges[c] = (start, end)
        print(f"  {c:<18} bytes {start:>11,} – {end:>11,}  ({(end-start)/1e6:6.1f} MB)", flush=True)
    total = sum(e - s for s, e in ranges.values())
    print(f"Downloading {total/1e6:.0f} MB instead of {remote.size/1e9:.2f} GB ...")

    parts_dir = out_rar.parent / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    def fetch(c):
        s, e = ranges[c]
        dest = parts_dir / f"{c}.bin"
        for attempt in range(30):
            have = dest.stat().st_size if dest.exists() else 0
            if have >= e - s:
                return c
            try:
                with requests.get(UCF101_URL, headers={"Range": f"bytes={s+have}-{e-1}"}, stream=True, timeout=60) as r:
                    if r.status_code != 206:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    with open(dest, "ab") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
            except (requests.RequestException, OSError, RuntimeError) as ex:
                print(f"  {c}: retry {attempt+1} ({ex})", flush=True)
                time.sleep(3)
        raise RuntimeError(f"failed to download range for {c}")

    def progress():
        while not done.is_set():
            have = sum((parts_dir / f"{c}.bin").stat().st_size for c in ranges if (parts_dir / f"{c}.bin").exists())
            print(f"  {have/1e6:7.1f} / {total/1e6:.1f} MB ({100*have/total:.1f}%)", flush=True)
            done.wait(10)

    import threading
    done = threading.Event()
    threading.Thread(target=progress, daemon=True).start()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fetch, ranges))
    done.set()

    with open(out_rar, "wb") as out:
        out.write(preamble)
        for c in sorted(ranges, key=lambda c: ranges[c][0]):
            with open(parts_dir / f"{c}.bin", "rb") as f:
                shutil.copyfileobj(f, out, 1 << 22)
        out.write(RAR_END)
    print(f"Reduced archive written: {out_rar} ({out_rar.stat().st_size/1e6:.0f} MB)")


def find_extractor() -> list[str] | None:
    candidates = [
        shutil.which("unrar"),
        r"C:\Program Files\WinRAR\UnRAR.exe",
        r"C:\Program Files (x86)\WinRAR\UnRAR.exe",
        shutil.which("7z"),
        r"C:\Program Files\7-Zip\7z.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return [c]
    return None


def extract_classes(archive: Path, classes: list[str]) -> None:
    exe = find_extractor()
    if exe is None:
        raise RuntimeError(
            "No RAR extractor found. Install WinRAR (UnRAR.exe) or 7-Zip, or extract "
            f"{archive} manually so that videos live in {UCF_DIR}/<ClassName>/*.avi"
        )
    UCF_DIR.parent.mkdir(parents=True, exist_ok=True)
    patterns = [f"UCF-101\\{c}\\*" if "7z" not in exe[0].lower() else f"UCF-101/{c}/*" for c in classes]
    if "7z" in exe[0].lower():
        cmd = exe + ["x", "-y", f"-o{UCF_DIR.parent}", str(archive)] + patterns
    else:
        cmd = exe + ["x", "-o+", "-inul", str(archive)] + patterns + [str(UCF_DIR.parent) + "\\"]
    print("Extracting:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    src = UCF_DIR.parent / "UCF-101"
    UCF_DIR.mkdir(parents=True, exist_ok=True)
    for c in classes:
        if (src / c).exists():
            if (UCF_DIR / c).exists():
                shutil.rmtree(UCF_DIR / c)
            shutil.move(str(src / c), str(UCF_DIR / c))
    if src.exists() and not any(src.iterdir()):
        src.rmdir()


def extract_splits(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(SPLITS_DIR.parent)


def validate(classes: list[str]) -> None:
    print("\nValidating class directories against the extracted dataset:")
    ok = True
    for c in classes:
        n = len(list((UCF_DIR / c).glob("*.avi")))
        print(f"  {c:<18} {n:>4} videos {'OK' if n else 'MISSING'}")
        ok &= n > 0
    idx = SPLITS_DIR / "classInd.txt"
    if idx.exists():
        names = {line.split()[1] for line in idx.read_text().splitlines() if line.strip()}
        bad = [c for c in classes if c not in names]
        if bad:
            ok = False
            print(f"  Class names not found in classInd.txt: {bad}")
    else:
        ok = False
        print(f"  Missing {idx}")
    if not ok:
        raise SystemExit("Validation failed — see messages above.")
    print("Dataset ready.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--classes", nargs="+", default=CLASSES)
    ap.add_argument("--full", action="store_true",
                    help="download the complete 6.9 GB archive instead of only the selected classes")
    args = ap.parse_args()

    full_rar = RAW_DIR / "UCF101.rar"
    subset_rar = RAW_DIR / "UCF101_subset.rar"
    splits_zip = RAW_DIR / "UCF101TrainTestSplits-RecognitionTask.zip"
    if not args.skip_download:
        print("Downloading official split files ...")
        if not splits_zip.exists():
            download(SPLITS_URL, splits_zip)
        extract_splits(splits_zip)
        if not full_rar.exists() and not subset_rar.exists():
            if args.full:
                print("Downloading full UCF101 (~6.9 GB). This can take a while ...")
                download(UCF101_URL, full_rar)
            else:
                try:
                    selective_download(args.classes, subset_rar)
                except Exception as e:
                    print(f"Selective download failed ({e}); falling back to the full archive.")
                    download(UCF101_URL, full_rar)
    rar = full_rar if full_rar.exists() else subset_rar
    for f in (rar, splits_zip):
        if not f.exists():
            raise SystemExit(f"Missing {f}.\n{MANUAL}")
    extract_splits(splits_zip)
    extract_classes(rar, args.classes)
    validate(args.classes)


if __name__ == "__main__":
    main()
