"""Phase-0 check: does the browser hand over media bodies?

Connects to your browser (remote debugging turned on at
chrome://inspect/#remote-debugging; allow the connection when it asks),
optionally opens URLs in it, and for a while logs every media response, whether
its body was retrievable, and what the StreamStore made of it. Run it while
playing a YouTube video, an HLS stream, and a plain MP4 page.

    python tests/manual/cdp_probe.py [--data "<User Data dir>"] [--seconds 60] [URL ...]

Pieces land in a temp directory, not the daemon's data/.
"""
import argparse
import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.sniffer.cdp import Sniffer  # noqa: E402
from core.sniffer.streams import StreamStore  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="the browser's User Data directory "
                                           "(default: Comet, Chrome, Edge, Brave)")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--keep", help="directory for captured pieces (default: temp)")
    parser.add_argument("urls", nargs="*")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    logging.getLogger("core.sniffer.cdp").setLevel(logging.DEBUG)

    root = Path(args.keep or tempfile.mkdtemp(prefix="cdp_probe_"))
    store = StreamStore(root)
    sniffer = Sniffer(store, data_dirs=[Path(args.data)]) if args.data else Sniffer(store)
    task = asyncio.create_task(sniffer.run())

    for _ in range(600):  # time to answer the browser's prompt
        if sniffer.connected:
            break
        await asyncio.sleep(0.2)
    if not sniffer.connected:
        print("Could not connect: is remote debugging on, and was the connection allowed?")
        return 1

    for url in args.urls:
        await sniffer.send("Target.createTarget", {"url": url})

    await asyncio.sleep(args.seconds)
    task.cancel()

    print(f"\nbodies={sniffer.stats['bodies']}  missed={sniffer.stats['missed']}  "
          f"bytes={sniffer.stats['bytes'] / 1e6:.1f} MB  -> {root}")
    for tab in sniffer.tabs.values():
        print(f"\ntab: {tab.title[:60]!r}  item={tab.item}")
    for item_dir in sorted(root.iterdir()):
        meta = json.loads((item_dir / "item.json").read_text())
        item = meta["item"]
        summary = store.summary(item)
        print(f"\nitem {item}: video={summary.has_video} drm={summary.drm} "
              f"streams={summary.streams} clear={summary.clear}")
        for line in store.progress(item):
            print("   ", line)
        for cand in store.candidates(item):
            print(f"    COMPLETE {cand.kind} {cand.bytes / 1e6:.1f} MB {cand.key[:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
