"""Headless test of the installed plug-in inside real GIMP. Environment: SMOKE_IN, SMOKE_OUT,
SMOKE_PRESET, SMOKE_MODEL, and SMOKE_RECT as x,y,w,h. Prints SMOKE ok or SMOKE fail.

    gimp-console-3.exe --no-interface --new-instance --batch-interpreter=python-fu-eval \\
        -b "exec(open(r'tools/gimp_smoke.py').read())" --quit
"""
import os
import time

import gi
gi.require_version('Gimp', '3.0')
from gi.repository import Gimp, Gio

try:
    x, y, w, h = (int(v) for v in os.environ["SMOKE_RECT"].split(","))
    image = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.File.new_for_path(os.environ["SMOKE_IN"]))
    layer = image.get_layers()[0]
    image.select_ellipse(Gimp.ChannelOps.REPLACE, x, y, w, h)

    procedure = Gimp.get_pdb().lookup_procedure("plug-in-moebius-inpaint")
    assert procedure is not None, "plug-in is not registered"
    config = procedure.create_config()
    config.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
    config.set_property("image", image)
    config.set_core_object_array("drawables", [layer])
    config.set_property("preset", os.environ["SMOKE_PRESET"])
    config.set_property("model", os.environ["SMOKE_MODEL"])
    config.set_property("seed", 0)

    started = time.monotonic()
    result = procedure.run(config)
    status = result.index(0)
    print(f"SMOKE status {status} in {time.monotonic() - started:.1f} s", flush=True)
    assert status == Gimp.PDBStatusType.SUCCESS, result.index(1)

    Gimp.Selection.none(image)
    Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, image, Gio.File.new_for_path(os.environ["SMOKE_OUT"]), None)
    print("SMOKE ok", flush=True)
except Exception as exc:
    print(f"SMOKE fail: {exc!r}", flush=True)
