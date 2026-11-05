#!/usr/bin/env python3
"""GIMP 3.2 plug-in: Moebius inpainting of the active selection.

GIMP's Python is a MINGW build that cannot load MSVC binaries, so inference runs out of
process: the layer and selection go to `moebius-cli.exe` as temporary PNGs, and the result
comes back through the shadow buffer, which also limits it to the selection. Only GIMP's
own modules are imported here.
"""

import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

import gi
gi.require_version('Gegl', '0.4')
gi.require_version('Gimp', '3.0')
gi.require_version('GimpUi', '3.0')

from gi.repository import Gegl, Gimp, GimpUi, GLib, GObject


PROCEDURE = "plug-in-moebius-inpaint"
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(PLUGIN_DIR, "moebius-cli.exe")
MODELS = {"scenes": "scenes", "faces": "faces"}
LOG_PATH = os.path.join(PLUGIN_DIR, "moebius.log")
# Exact takes ~25 s on the benchmark CPU; leave room for slower ones.
TIMEOUT_SECONDS = 900
POLL_SECONDS = 0.1
MAX_SEED = 2**31 - 1
# Share of the progress bar each stage ends at.
ENCODE_END, DENOISE_END = 0.10, 0.92


def log(message):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass


def save_png(buffer, path):
    graph = Gegl.Node()
    source = graph.create_child("gegl:buffer-source")
    source.set_property("buffer", buffer)
    saver = graph.create_child("gegl:png-save")
    saver.set_property("path", path)
    saver.set_property("bitdepth", 8)
    saver.set_property("compression", 1)
    source.link(saver)
    saver.process()
    if not os.path.isfile(path):
        raise OSError(f"GEGL did not create {path}")


def save_selection(image, drawable, path):
    """Selection in layer coordinates, as an 8-bit mask."""
    success, x, y = drawable.get_offsets()
    if not success:
        raise RuntimeError("Could not read the layer offset")
    width, height = drawable.get_width(), drawable.get_height()
    local = Gegl.Buffer.new("Y u8", 0, 0, width, height)
    image.get_selection().get_buffer().copy(
        Gegl.Rectangle.new(x, y, width, height),
        Gegl.AbyssPolicy.NONE,
        local,
        Gegl.Rectangle.new(0, 0, width, height),
    )
    save_png(local, path)


def load_into_shadow(drawable, path):
    shadow = drawable.get_shadow_buffer()
    graph = Gegl.Node()
    loader = graph.create_child("gegl:png-load")
    loader.set_property("path", path)
    bounds = loader.get_bounding_box()
    expected = (drawable.get_width(), drawable.get_height())
    if (bounds.width, bounds.height) != expected:
        raise ValueError(f"result is {bounds.width}x{bounds.height}, layer is {expected[0]}x{expected[1]}")
    writer = graph.create_child("gegl:write-buffer")
    writer.set_property("buffer", shadow)
    loader.link(writer)
    writer.process()
    shadow.flush()


def fraction(stage, done, total):
    share = done / total if total else 0.0
    if stage == "encode":
        return ENCODE_END * share
    if stage == "denoise":
        return ENCODE_END + (DENOISE_END - ENCODE_END) * share
    return DENOISE_END + (1.0 - DENOISE_END) * share


def run_worker(command, on_progress):
    """Runs the worker, returning (exit code, its other output). None on timeout.

    stdin stays open for the run: the worker exits when it closes, so killing this
    plug-in (GIMP's progress cancel button) cannot leave a worker behind.
    """
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=flags,
    )
    lines = queue.Queue()

    def reader():
        for line in iter(process.stdout.readline, ""):
            lines.put(line)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    other = []
    deadline = time.monotonic() + TIMEOUT_SECONDS

    def drain():
        while True:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                return
            parts = line.split()
            if len(parts) == 4 and parts[0] == "PROGRESS":
                on_progress(parts[1], int(parts[2]), int(parts[3]))
            elif line.strip():
                other.append(line.strip())

    try:
        while process.poll() is None:
            drain()
            if time.monotonic() > deadline:
                return None, ""
            time.sleep(POLL_SECONDS)
        thread.join(timeout=2.0)
        drain()
        return process.returncode, "\n".join(other[-8:])
    finally:
        # Windows keeps the exe locked while any handle to the child is open.
        if process.poll() is None:
            process.kill()
        process.wait()
        for pipe in (process.stdin, process.stdout):
            try:
                pipe.close()
            except OSError:
                pass


def error(procedure, status, message):
    log(f"error: {message}")
    return procedure.new_return_values(status, GLib.Error.new_literal(Gimp.PlugIn.error_quark(), message, 0))


class MoebiusInpaint(Gimp.PlugIn):
    def do_set_i18n(self, _name):
        return False

    def do_query_procedures(self):
        return [PROCEDURE]

    def do_create_procedure(self, name):
        if name != PROCEDURE:
            return None
        Gegl.init(None)
        procedure = Gimp.ImageProcedure.new(self, name, Gimp.PDBProcType.PLUGIN, self.run, None)
        procedure.set_image_types("RGB*, GRAY*")
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
        procedure.set_menu_label("_Moebius Inpaint...")
        procedure.set_icon_name(GimpUi.ICON_GEGL)
        procedure.add_menu_path("<Image>/Filters/Enhance/")
        procedure.set_documentation(
            "Remove the selected content with the Moebius diffusion model.",
            "Runs on the CPU. The selection must cover the whole object, shadows included, "
            "with a margin: anything left outside it can grow back. Only the selection changes.",
            name,
        )
        procedure.set_attribution("moebius-cpu", "moebius-cpu", "2026")

        preset = Gimp.Choice.new()
        preset.add("fast", 0, "Fast", "")
        preset.add("exact", 1, "Exact (about twice as slow)", "")
        procedure.add_choice_argument(
            "preset", "_Quality", "Fast is close to Exact in about half the time",
            preset, "fast", GObject.ParamFlags.READWRITE,
        )
        model = Gimp.Choice.new()
        model.add("scenes", 0, "Scenes and objects", "")
        model.add("faces", 1, "Faces", "")
        procedure.add_choice_argument(
            "model", "_Content", "Checkpoint matching what the picture shows",
            model, "scenes", GObject.ParamFlags.READWRITE,
        )
        procedure.add_int_argument(
            "seed", "_Seed", "Change this to get a different result",
            0, MAX_SEED, 0, GObject.ParamFlags.READWRITE,
        )
        return procedure

    def run(self, procedure, run_mode, image, drawables, config, _data):
        if run_mode == Gimp.RunMode.INTERACTIVE:
            GimpUi.init("moebius-inpaint")
            dialog = GimpUi.ProcedureDialog.new(procedure, config, "Moebius Inpaint")
            dialog.fill(["preset", "model", "seed"])
            accepted = dialog.run()
            dialog.destroy()
            if not accepted:
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())
        try:
            return self.inpaint(
                procedure, image, drawables,
                config.get_property("preset"), config.get_property("model"), config.get_property("seed"),
            )
        except Exception as exc:
            return error(procedure, Gimp.PDBStatusType.EXECUTION_ERROR, f"Moebius Inpaint failed: {exc}")

    def inpaint(self, procedure, image, drawables, preset, model, seed):
        calling = Gimp.PDBStatusType.CALLING_ERROR
        if len(drawables) != 1:
            return error(procedure, calling, "Select exactly one layer.")
        drawable = drawables[0]
        intersects, x, y, width, height = drawable.mask_intersect()
        if not intersects or Gimp.Selection.is_empty(image):
            return error(procedure, calling, "Select what you want removed first.")
        weights = os.path.join(PLUGIN_DIR, "models", MODELS[model])
        for path in (WORKER, os.path.join(weights, "unet.safetensors"), os.path.join(weights, "vae.safetensors")):
            if not os.path.isfile(path):
                return error(procedure, calling, f"Missing {path}. Run install.bat again.")

        Gimp.progress_init("Moebius Inpaint")
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="gimp-moebius-") as temp:
                source = os.path.join(temp, "image.png")
                mask = os.path.join(temp, "mask.png")
                result = os.path.join(temp, "result.png")
                Gimp.progress_set_text("Preparing the layer...")
                save_png(drawable.get_buffer(), source)
                save_selection(image, drawable, mask)

                labels = {"encode": "Reading the image...", "denoise": "Inpainting...", "decode": "Finishing..."}

                def on_progress(stage, done, total):
                    Gimp.progress_set_text(labels[stage])
                    Gimp.progress_update(fraction(stage, done, total))

                code, output = run_worker(
                    [WORKER, "--plugin", "--weights", weights, "--image", source, "--mask", mask,
                     "--output", result, "--preset", preset, "--seed", str(seed)],
                    on_progress,
                )
                if code is None:
                    return error(procedure, Gimp.PDBStatusType.EXECUTION_ERROR, f"Timed out after {TIMEOUT_SECONDS} s.")
                if code != 0 or not os.path.isfile(result):
                    return error(procedure, Gimp.PDBStatusType.EXECUTION_ERROR, f"The worker failed ({code}): {output}")
                load_into_shadow(drawable, result)

            drawable.merge_shadow(True)
            drawable.update(x, y, width, height)
            Gimp.displays_flush()
            log(f"ok: {preset} {model} seed {seed}, {time.monotonic() - started:.1f} s")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
        finally:
            Gimp.progress_end()


Gimp.main(MoebiusInpaint.__gtype__, sys.argv)
