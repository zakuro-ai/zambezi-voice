#!/usr/bin/env python3
"""Package Zambezi Voice for the Zakuro hub and emit a La Forge ASR manifest.

  build     pack each (language, split)'s audio into tar shards and write
            manifest.csv, the La Forge manifest: one row per clip, audio -> transcript
  push      upload a directory as a (private) dataset version on the hub
  verify    fetch clips back from the hub by byte range and check their sha256

Why shards: a hub dataset version holds at most 256 files (zak-marketplace
api/manifests.py MAX_FILES) and this repository has ~12k WAVs. A tar member's
bytes are stored contiguously and unmodified, so a clip stays addressable as
(shard, offset, size): an HTTP Range request for exactly that span returns the
original .wav, byte for byte.

Standard library only. The hub token must be a web session (a `zc login` CLI
token is refused by the dataset routes); it is read from $ZAKURO_HUB_TOKEN or
--token-file and never printed.
"""
from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures as cf
import csv
import hashlib
import http.client
import io
import json
import os
import random
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

MiB = 1024 * 1024
#: Mirrors zak-marketplace api/multipart.py: one PUT up to 64 MiB, 32 MiB
#: parts above it.
MULTIPART_THRESHOLD = 64 * MiB
PART_SIZE = 32 * MiB
#: zak-marketplace api/manifests.py MAX_FILES.
MAX_FILES = 256
SPLITS = ("train", "dev", "test")
DEFAULT_HUB = "https://stg.hub.zakuro-ai.com"
DEFAULT_TOKEN_FILE = "~/.config/zakuro/stg-hub.token"
USER_AGENT = "zambezi-voice-hub/1.0"
UPSTREAM = "https://github.com/zakuro-ai/zambezi-voice"

#: The La Forge manifest: the dataset's only CSV. `audio` is relative to the
#: dataset version (`<shard>#<member>`), so the manifest travels with the
#: audio it names and never has to know its own version digest.
MANIFEST_FIELDS = ["audio", "transcript", "split", "language", "duration_ms",
                   "sample_rate", "offset", "size", "sha256"]


class HubError(Exception):
    pass


# ---------------------------------------------------------------------- build

def _tarinfo(name: str, size: int) -> tarfile.TarInfo:
    """A member header with every varying field pinned, so rebuilding from the
    same checkout yields byte-identical shards -- and the same hub digest."""
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


class ShardWriter:
    """Writes `<split>-NNN.tar`, starting a new shard before the current one
    would grow past `max_bytes`."""

    def __init__(self, root: Path, rel_dir: str, split: str, max_bytes: int):
        self.root, self.rel_dir, self.split = root, rel_dir, split
        self.max_bytes = max_bytes
        self.number = -1
        self.tar: tarfile.TarFile | None = None
        self.rel = ""
        self.count = 0

    def _roll(self) -> None:
        self.close()
        self.number += 1
        self.rel = f"{self.rel_dir}/{self.split}-{self.number:03d}.tar"
        path = self.root / self.rel
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tar = tarfile.open(path, "w", format=tarfile.USTAR_FORMAT)
        self.count = 0

    def add(self, name: str, data: bytes) -> tuple[str, int]:
        """Append one member and return (shard path, offset of its bytes)."""
        padded = -(-len(data) // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE
        if self.tar is None or (self.count and self.tar.offset
                                + tarfile.BLOCKSIZE + padded > self.max_bytes):
            self._roll()
        info = _tarinfo(name, len(data))
        header = info.tobuf(self.tar.format, self.tar.encoding, self.tar.errors)
        offset = self.tar.offset + len(header)
        self.tar.addfile(info, io.BytesIO(data))
        self.count += 1
        return self.rel, offset

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def wav_facts(data: bytes) -> tuple[int, int] | None:
    """(duration_ms, sample_rate) from the RIFF header, or None."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            rate = w.getframerate()
            return round(w.getnframes() * 1000 / rate), rate
    except (wave.Error, EOFError, ZeroDivisionError):
        return None


def verify_shards(root: Path, rows: list[dict]) -> None:
    """Read every clip back out of its shard two independent ways: tarfile's
    own parser must agree on (offset, size), and the bytes at that span must
    hash to the source file's sha256."""
    by_shard = collections.defaultdict(list)
    for r in rows:
        by_shard[r["shard"]].append(r)
    for shard, items in sorted(by_shard.items()):
        with tarfile.open(root / shard) as t:
            members = {m.name: m for m in t.getmembers()}
        if len(members) != len(items):
            raise SystemExit(f"{shard}: {len(members)} members but "
                             f"{len(items)} indexed clips")
        with (root / shard).open("rb") as f:
            for r in items:
                m = members[r["member"]]
                if (m.offset_data, m.size) != (r["offset"], r["size"]):
                    raise SystemExit(
                        f"{shard}#{m.name}: tar says {m.offset_data}+{m.size}, "
                        f"index says {r['offset']}+{r['size']}")
                f.seek(r["offset"])
                if hashlib.sha256(f.read(r["size"])).hexdigest() != r["sha256"]:
                    raise SystemExit(f"{shard}#{m.name}: bytes at the indexed "
                                     "span do not match the source file")


def write_card(root: Path, stats: dict, skipped: list[str],
               problems: collections.Counter) -> None:
    table = ["| language | code | split | clips | hours |",
             "|---|---|---|---:|---:|"]
    clips = ms = 0
    for (language, code, split), (n, t) in sorted(stats.items()):
        table.append(f"| {language} | `{code}` | {split} | {n:,} | {t / 3.6e6:.2f} |")
        clips, ms = clips + n, ms + t
    table.append(f"| **total** | | | **{clips:,}** | **{ms / 3.6e6:.2f}** |")
    omitted = [f"- {n:,} × {what}" for what, n in sorted(problems.items()) if n]
    skipped_md = [f"- `{rel}`: transcripts only; its audio is not in this "
                  "repository (Nyanja's lives in "
                  "[unza-speech-lab/zambezi-voice-nyanja]"
                  "(https://github.com/unza-speech-lab/zambezi-voice-nyanja))."
                  for rel in skipped]
    card = f"""# Zambezi Voice: read speech for ASR

Read-speech recordings with transcripts from the
[Zambezi Voice](https://github.com/unza-speech-lab/zambezi-voice) project
(University of Zambia Speech and Language Research Group; Sikasote et al.,
[Interspeech 2023](https://www.isca-speech.org/archive/pdfs/interspeech_2023/sikasote23_interspeech.pdf)),
packaged from [zakuro-ai/zambezi-voice]({UPSTREAM}).

{chr(10).join(table)}

## Layout

- `manifest.csv`: one row per clip. `audio` names the clip as
  `<shard>#<member>`, relative to this dataset; `offset`, `size` and `sha256`
  locate and check its bytes inside the shard; then `transcript`, `split`,
  `language` (ISO 639-3), `duration_ms`, `sample_rate`. It is the dataset's
  only CSV, which is what a La Forge `dataset_ref` resolves to.
- `<language>/<code>/audio/<split>-NNN.tar`: the audio (16 kHz mono 16-bit
  PCM WAV), in uncompressed tar shards because a hub dataset version holds
  at most 256 files.

A clip's bytes sit contiguously and unmodified in its shard, so one HTTP Range
request (`bytes=<offset>-<offset + size - 1>`) returns the original `.wav`.
The original TSVs are in the repository, not here, so the catalogue counts
each clip once.

## Not included

{chr(10).join(skipped_md + omitted) or "- nothing"}

## License

MIT, as declared by the upstream repository's `LICENSE`. Please cite the
Zambezi Voice paper when you use this data.
"""
    (root / "README.md").write_text(card, encoding="utf-8")


def cmd_build(args) -> None:
    repo, out = Path(args.repo).resolve(), Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; remove it so no stale shard "
                         "gets uploaded")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped: list[str] = []
    problems: collections.Counter = collections.Counter()
    stats: dict = collections.defaultdict(lambda: [0, 0])
    for code_dir in sorted(p.parent for p in repo.glob("*/*/train.tsv")):
        rel = code_dir.relative_to(repo).as_posix()
        audio_dir = code_dir / "audio"
        if not audio_dir.is_dir():
            skipped.append(rel)
            continue
        language, code = code_dir.parent.name.capitalize(), code_dir.name
        on_disk = {p.name for p in audio_dir.glob("*.wav")}
        seen: set[str] = set()
        for split in SPLITS:
            tsv = code_dir / f"{split}.tsv"
            if not tsv.exists():
                continue
            writer = ShardWriter(out, f"{rel}/audio", split, args.shard_bytes)
            with tsv.open(newline="", encoding="utf-8") as f:
                for rec in csv.DictReader(f, delimiter="\t"):
                    audio_id = (rec.get("audio_id") or "").strip()
                    transcript = (rec.get("sentence") or "").strip()
                    if audio_id not in on_disk:
                        problems["transcript rows whose audio file is missing"] += 1
                        continue
                    if not transcript:
                        problems["clips with an empty transcript"] += 1
                        continue
                    if audio_id in seen:
                        problems["clips listed in more than one split (kept)"] += 1
                    seen.add(audio_id)
                    data = (audio_dir / audio_id).read_bytes()
                    facts = wav_facts(data)
                    if facts is None:
                        problems["unreadable WAV headers (TSV duration kept)"] += 1
                        facts = (int(float(rec["durationMsec"])),
                                 int(rec["sampleRate"]))
                    elif abs(facts[0] - float(rec.get("durationMsec") or 0)) \
                            > max(20, 0.01 * facts[0]):
                        problems["TSV durations that disagree with the WAV (WAV kept)"] += 1
                    shard, offset = writer.add(audio_id, data)
                    rows.append({
                        "language": code, "split": split, "audio_id": audio_id,
                        "transcript": transcript, "duration_ms": facts[0],
                        "sample_rate": facts[1], "shard": shard,
                        "member": audio_id, "offset": offset, "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
                    stats[(language, code, split)][0] += 1
                    stats[(language, code, split)][1] += facts[0]
            writer.close()
        problems["audio files no transcript references"] += len(on_disk - seen)

    verify_shards(out, rows)
    write_card(out, stats, skipped, problems)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({"audio": f"{r['shard']}#{r['member']}",
                        "transcript": r["transcript"], "split": r["split"],
                        "language": r["language"], "duration_ms": r["duration_ms"],
                        "sample_rate": r["sample_rate"], "offset": r["offset"],
                        "size": r["size"], "sha256": r["sha256"]})

    files = sorted(p for p in out.rglob("*") if p.is_file())
    print(f"{len(rows):,} clips -> {len(files)} files, "
          f"{sum(p.stat().st_size for p in files) / 1e9:.2f} GB in {out}")
    for p in files:
        print(f"  {p.stat().st_size / MiB:9.1f} MiB  {p.relative_to(out)}")
    for rel in skipped:
        print(f"skipped {rel}: no audio/ directory")
    for what, n in sorted(problems.items()):
        if n:
            print(f"note: {n:,} {what}")
    print(f"manifest: {out / 'manifest.csv'}")


# ----------------------------------------------------------------------- hub

def load_token(args) -> str:
    token = os.environ.get("ZAKURO_HUB_TOKEN", "").strip()
    if token:
        return token
    path = Path(args.token_file).expanduser()
    if not path.is_file():
        raise SystemExit(f"no hub token: set ZAKURO_HUB_TOKEN or write one to {path}")
    return path.read_text().strip()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Hub:
    def __init__(self, base: str, token: str):
        self.base = base.rstrip("/")
        self._token = token
        self._no_redirect = urllib.request.build_opener(_NoRedirect)

    def _request(self, method: str, url: str, body=None) -> urllib.request.Request:
        if not url.startswith(self.base + "/"):
            raise HubError(f"refusing to send the hub token to {url}")
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("User-Agent", USER_AGENT)
        req.add_header("Authorization", f"Bearer {self._token}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        return req

    def call(self, method: str, path: str, body=None):
        req = self._request(method, self.base + path, body)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:1500]
            raise HubError(f"{method} {path} -> {e.code}: {detail}") from None
        return json.loads(raw) if raw else None

    def location(self, url: str) -> str:
        """The presigned storage URL a hub file route redirects to. Followed by
        hand so the bearer token is never replayed to the storage host."""
        try:
            with self._no_redirect.open(self._request("GET", url), timeout=60) as r:
                raise HubError(f"GET {url} -> {r.status}, expected a redirect")
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                return e.headers["Location"]
            detail = e.read().decode(errors="replace")[:500]
            raise HubError(f"GET {url} -> {e.code}: {detail}") from None


def hash_file(path: Path, with_parts: bool) -> tuple[str, list[str]]:
    whole, parts = hashlib.sha256(), []
    with path.open("rb") as f:
        while chunk := f.read(PART_SIZE):
            whole.update(chunk)
            if with_parts:
                parts.append(hashlib.sha256(chunk).hexdigest())
    return whole.hexdigest(), parts


def tabular_meta(path: Path) -> dict | None:
    """The row count, plus a CSV's header. A TSV declares no columns: the hub
    reads a header as comma-separated and would refuse a tab-separated one."""
    if path.suffix not in (".csv", ".tsv"):
        return None
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="," if path.suffix == ".csv" else "\t")
        header = next(reader, None)
        records = sum(1 for _ in reader)
    meta: dict = {"records": records}
    if path.suffix == ".csv" and header:
        meta["columns"] = [c.strip() for c in header]
    return meta


def declare(root: Path) -> list[dict]:
    """The upload manifest for every non-hidden file under `root`."""
    files = sorted((p for p in root.rglob("*")
                    if p.is_file() and not p.name.startswith(".")),
                   key=lambda p: p.relative_to(root).as_posix().encode())
    if not files:
        raise SystemExit(f"nothing to upload in {root}")
    if len(files) > MAX_FILES:
        raise SystemExit(f"{len(files)} files; a hub version holds at most {MAX_FILES}")
    out = []
    for p in files:
        size = p.stat().st_size
        multipart = size > MULTIPART_THRESHOLD
        sha, parts = hash_file(p, multipart)
        entry: dict = {"path": p.relative_to(root).as_posix(), "sha256": sha,
                       "size_bytes": size}
        if multipart:
            entry["part_size"] = PART_SIZE
            entry["parts"] = [{"part_number": i + 1, "sha256": h}
                              for i, h in enumerate(parts)]
        meta = tabular_meta(p)
        if meta:
            entry["meta"] = meta
        out.append(entry)
    return out


def set_digest(files: list[dict]) -> str:
    """zak-marketplace api/manifests.py set_digest: sha256 over one
    "<sha256>  <path>\\n" line per file, paths sorted as bytes."""
    rows = sorted(((f["path"], f["sha256"]) for f in files),
                  key=lambda e: e[0].encode())
    return hashlib.sha256("".join(f"{s}  {p}\n" for p, s in rows).encode()).hexdigest()


def put(url: str, data: bytes, sha_hex: str) -> str | None:
    """PUT bytes to a presigned URL bound to their sha256; return the ETag."""
    checksum = base64.b64encode(bytes.fromhex(sha_hex)).decode()
    error = ""
    for attempt in range(6):
        req = urllib.request.Request(url, data=data, method="PUT", headers={
            "x-amz-checksum-sha256": checksum,
            "Content-Type": "application/octet-stream",
            "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return r.headers.get("ETag")
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            if e.code < 500 and e.code not in (408, 429):
                raise HubError(f"storage refused a PUT ({e.code}): {body}") from None
            error = f"{e.code} {body}"
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            error = repr(e)
        time.sleep(min(60, 2 ** attempt))
    raise HubError(f"PUT failed after retries: {error}")


def cmd_push(args) -> None:
    root = Path(args.dir).resolve()
    hub = Hub(args.hub, load_token(args))
    me = hub.call("GET", "/api/accounts/me")

    files = declare(root)
    digest = set_digest(files)
    total = sum(f["size_bytes"] for f in files)
    print(f"{args.name}: {len(files)} files, {total / 1e9:.2f} GB, "
          f"digest {digest}, as {me['username']}", flush=True)

    opened = hub.call("POST", "/api/datasets/uploads", {
        "name": args.name, "visibility": args.visibility,
        "license": args.license, "files": files})
    if opened["digest"] != digest:
        raise HubError(f"hub computed digest {opened['digest']}, local {digest}")
    upload_id, dataset_id = opened["upload_id"], opened["dataset_id"]

    state_path = root / ".hub-push-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    etags = state.setdefault(upload_id, {})
    lock = threading.Lock()
    by_path = {f["path"]: f for f in files}

    jobs = []
    for pf in opened["files"]:
        path = root / pf["path"]
        if "url" in pf:
            jobs.append((pf["path"], None, pf["url"], 0,
                         by_path[pf["path"]]["size_bytes"],
                         by_path[pf["path"]]["sha256"]))
            continue
        declared = {p["part_number"]: p["sha256"] for p in by_path[pf["path"]]["parts"]}
        for part in pf["parts"]:
            n = part["part_number"]
            offset = (n - 1) * pf["part_size"]
            length = min(pf["part_size"], path.stat().st_size - offset)
            jobs.append((pf["path"], n, part["url"], offset, length, declared[n]))

    todo = sum(j[4] for j in jobs)
    done, last = 0, 0.0
    started = time.monotonic()

    def run(job):
        rel, n, url, offset, length, sha = job
        with (root / rel).open("rb") as f:
            f.seek(offset)
            data = f.read(length)
        etag = put(url, data, sha)
        if n is not None:
            if etag is None:
                raise HubError(f"part {n} of {rel} landed without an ETag")
            with lock:
                etags.setdefault(rel, {})[str(n)] = etag
                state_path.write_text(json.dumps(state))
        return length

    print(f"uploading {len(jobs)} objects/parts, {todo / 1e9:.2f} GB "
          f"({(total - todo) / 1e9:.2f} GB already landed)", flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in cf.as_completed([pool.submit(run, j) for j in jobs]):
            done += fut.result()
            now = time.monotonic()
            if now - last > 10 or done == todo:
                rate = done / max(now - started, 1e-6) / MiB
                print(f"  {done / 1e9:6.2f} / {todo / 1e9:.2f} GB  "
                      f"{rate:6.1f} MiB/s", flush=True)
                last = now

    ready = opened.get("expires_at") is None and not opened["files"]
    if not ready:
        # Assemble only the multipart files still in flight in THIS upload.
        # The hub stores blobs by content, so a file whose bytes it already
        # holds (from an earlier version, or assembled by an earlier run)
        # reports landed and was never opened as multipart here: there are
        # no parts to complete and no ETags to have.
        status = hub.call("GET", f"/api/datasets/uploads/{upload_id}")
        landed = {f["path"] for f in status["files"] if f["landed"]}
        for f in files:
            if "parts" not in f or f["path"] in landed:
                continue
            have = etags.get(f["path"], {})
            if len(have) != len(f["parts"]):
                raise HubError(
                    f"{f['path']}: have ETags for {len(have)} of {len(f['parts'])} "
                    f"parts; abandon upload {upload_id} "
                    f"(DELETE /api/datasets/uploads/{upload_id}) and push again")
            quoted = urllib.parse.quote(f["path"])
            hub.call("POST", f"/api/datasets/uploads/{upload_id}/files/{quoted}/complete",
                     {"parts": [{"part_number": int(n), "etag": e}
                                for n, e in sorted(have.items(), key=lambda kv: int(kv[0]))]})
        version = hub.call("POST", f"/api/datasets/uploads/{upload_id}/complete", {})
    else:
        version = hub.call("GET", f"/api/datasets/{dataset_id}/versions/{digest}")

    # `zc://` names the owner by `users.handle`, which /api/accounts/me does
    # not carry; the dataset view does.
    handle = hub.call("GET", f"/api/datasets/{dataset_id}").get("owner_handle")
    if not handle:
        raise HubError(f"dataset {dataset_id} came back without an owner_handle")
    ref = f"zc://{handle}/{args.name}@sha256:{version['digest']}"
    resolved = hub.call("GET", "/api/datasets/resolve?ref=" + urllib.parse.quote(ref, safe=""))
    if resolved["dataset_id"] != dataset_id or resolved["digest"] != digest:
        raise HubError(f"{ref} resolved to {resolved}")
    result = {
        "hub": hub.base, "handle": handle, "name": args.name,
        "visibility": args.visibility, "dataset_id": dataset_id,
        "version_id": version["id"], "digest": version["digest"],
        "status": version["status"], "ref": ref, "size_bytes": version["size_bytes"],
        "paths": [f["path"] for f in version["files"]],
        "meta": {k: v for k, v in (version.get("meta") or {}).items()
                 if k != "card_markdown"},
    }
    Path(args.result).write_text(json.dumps(result, indent=2) + "\n")
    state_path.unlink(missing_ok=True)
    print(f"published {ref} ({version['status']}), dataset {dataset_id}")
    print(f"result: {args.result}")


def cmd_verify(args) -> None:
    pushed = json.loads(Path(args.hub_result).read_text())
    hub = Hub(pushed["hub"], load_token(args))
    got = hub.call("GET", "/api/datasets/resolve?ref="
                   + urllib.parse.quote(pushed["ref"], safe=""))
    if got["canonical"] != pushed["ref"]:
        raise SystemExit(f"{pushed['ref']} resolves to {got['canonical']}")
    print(f"resolves: {pushed['ref']}")
    files_base = (f"{hub.base}/api/datasets/{pushed['dataset_id']}"
                  f"/versions/{pushed['digest']}/files/")
    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_shard = collections.defaultdict(list)
    for r in rows:
        shard, _, member = r["audio"].partition("#")
        by_shard[shard].append((member, r))
    rng = random.Random(args.seed)
    checked = 0
    for shard, items in sorted(by_shard.items()):
        picks = range(len(items)) if args.all else sorted(
            {0, len(items) - 1} | set(rng.sample(range(len(items)),
                                                 min(args.per_shard, len(items)))))
        storage = hub.location(files_base + urllib.parse.quote(shard))
        for i in picks:
            member, r = items[i]
            start, size = int(r["offset"]), int(r["size"])
            req = urllib.request.Request(storage, headers={
                "Range": f"bytes={start}-{start + size - 1}", "User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                status, data = resp.status, resp.read()
            if (status != 206 or len(data) != size
                    or hashlib.sha256(data).hexdigest() != r["sha256"]):
                raise SystemExit(f"{shard}#{member}: HTTP {status}, {len(data)} bytes, "
                                 "short read or sha256 mismatch")
            facts = wav_facts(data)
            if facts is None or facts[1] != int(r["sample_rate"]):
                raise SystemExit(f"{shard}#{member}: not the WAV the manifest describes")
            checked += 1
        print(f"  ok {len(picks):4d} clips  {shard}", flush=True)
    print(f"verified {checked} clips across {len(by_shard)} shards by byte-range fetch")


# ----------------------------------------------------------------------- cli

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="pack audio into shards and write the manifest")
    b.add_argument("--repo", default=".")
    b.add_argument("--out", default="build/hub/zambezi-voice")
    b.add_argument("--shard-bytes", type=int, default=512 * MiB)
    b.set_defaults(fn=cmd_build)

    p = sub.add_parser("push", help="upload a directory as a dataset")
    p.add_argument("--hub", default=DEFAULT_HUB)
    p.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    p.add_argument("dir")
    p.add_argument("--name", required=True)
    p.add_argument("--visibility", choices=("private", "public"), default="private")
    p.add_argument("--license", default="mit")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--result", required=True)
    p.set_defaults(fn=cmd_push)

    v = sub.add_parser("verify", help="range-fetch clips from the hub and check them")
    v.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    v.add_argument("--hub-result", default="build/zambezi-voice.hub.json")
    v.add_argument("--manifest", default="build/hub/zambezi-voice/manifest.csv")
    v.add_argument("--per-shard", type=int, default=3)
    v.add_argument("--all", action="store_true")
    v.add_argument("--seed", type=int, default=0)
    v.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except HubError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
