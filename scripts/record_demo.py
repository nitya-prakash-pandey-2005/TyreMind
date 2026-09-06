"""Render the self-driving demo to a 4:55 MP4.

Playwright drives a real Chromium at 1920x1080 and records the page; the demo
page drives itself. Three details matter:

  * color_scheme="dark" -- headless Chromium reports a light preference, and the
    app honours it, which puts a washed-out light UI under a dark caption dock.
  * The capture necessarily includes the boot screen, whose length varies with
    how fast the session fits. Rather than guess an offset, the start is found by
    measuring the caption band: it is flat dark while the gate is up and contains
    white text the moment the timeline begins.
  * yuv420p and +faststart, so Google Drive previews it inline instead of asking
    anyone to download it first.

    python record.py <out.mp4>
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
from PIL import Image
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8077/demo.html"
W, H = 1920, 1080
RUNTIME = 295.0          # the demo's own timeline length, in seconds
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "TyreMind_Demo.mp4")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def duration(path: Path) -> float:
    err = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def caption_band_lit(path: Path, t: float, tmp: Path) -> bool:
    """True once the caption dock carries white text, i.e. the timeline started."""
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path),
         "-frames:v", "1", str(tmp)],
        capture_output=True,
    )
    if not tmp.exists():
        return False
    im = Image.open(tmp).convert("L")
    band = im.crop((0, im.height - 120, im.width, im.height))
    return band.getextrema()[1] > 170


def find_start(path: Path, tmp: Path) -> float:
    """Coarse scan then a bisect, so the trim is not a guess."""
    lo, hi = 0.0, 1.0
    for t in [x / 2 for x in range(0, 61)]:          # 0 .. 30 s in 0.5 s steps
        if caption_band_lit(path, t, tmp):
            hi = t
            lo = max(0.0, t - 0.5)
            break
    else:
        raise SystemExit("could not find where the timeline starts")
    for _ in range(5):                                # bisect to ~0.03 s
        mid = (lo + hi) / 2
        if caption_band_lit(path, mid, tmp):
            hi = mid
        else:
            lo = mid
    return hi


def main() -> int:
    raw = OUT.with_name(OUT.stem + "_raw.webm")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--force-device-scale-factor=1", "--hide-scrollbars", "--disable-gpu",
                  "--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--mute-audio"],
        )
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(OUT.parent),
            record_video_size={"width": W, "height": H},
            device_scale_factor=1,
            color_scheme="dark",
        )
        page = ctx.new_page()
        print(f"  loading {URL}")
        page.goto(URL, wait_until="load", timeout=60_000)

        print("  waiting for the app to finish booting")
        page.wait_for_function("document.body.dataset.ready === '1'", timeout=180_000)
        time.sleep(1.5)

        print("  running the timeline")
        page.evaluate("window.startDemo()")
        page.wait_for_function("document.body.dataset.done === '1'", timeout=400_000)
        time.sleep(2.5)

        video = page.video
        ctx.close()
        browser.close()
        src = Path(video.path())

    if raw.exists():
        raw.unlink()
    src.rename(raw)
    print(f"  raw capture {raw.name} ({raw.stat().st_size / 1e6:.1f} MB, {duration(raw):.1f}s)")

    tmp = OUT.with_name("_probe.png")
    start = find_start(raw, tmp)
    tmp.unlink(missing_ok=True)
    print(f"  timeline starts at {start:.2f}s — trimming {RUNTIME:.0f}s from there")

    if OUT.exists():
        OUT.unlink()
    res = subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-accurate_seek", "-ss", f"{start:.2f}", "-i", str(raw),
         "-t", f"{RUNTIME}", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "30", str(OUT)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(res.stderr[-2500:])
        return 1

    raw.unlink(missing_ok=True)
    d = duration(OUT)
    print(f"\n  DONE  {OUT}")
    print(f"        {d // 60:.0f}:{d % 60:05.2f}  ·  {OUT.stat().st_size / 1e6:.1f} MB  ·  1920x1080 H.264")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
