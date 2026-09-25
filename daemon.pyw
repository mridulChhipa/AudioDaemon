"""Invisible launcher. Double-click, or drop a shortcut in shell:startup.

All the logic lives in daemon.py so `python daemon.py` stays usable for
debugging with a visible console.
"""
from daemon import main

raise SystemExit(main())
