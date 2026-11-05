#!/usr/bin/env python3
"""Installs Moebius Inpaint for the current user's GIMP 3.2.

Copies the plug-in and its compiled programs into GIMP's plug-ins folder, downloads the models
from Hugging Face and converts them with moebius-convert. It uses only the standard library, so
GIMP's own Python can run it: install.bat finds that Python. Running it again resumes a download
and skips finished models.

    python install.py [--weights DIR] [--dest DIR]

--weights, for developers, is a folder with ft_places2/ and ft_celebahq/ already converted.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# Pinned upstream revisions. moebius-convert checks each file against these hashes.
MOEBIUS_URL = "https://huggingface.co/hustvl/Moebius/resolve/cd01f47fb648219d3fa605806c5ce00b713faa5e"
VAE_URL = "https://huggingface.co/hustvl/PixelHacker/resolve/012fd343158936a265b8a0ee38a791a7a2841f45/vae"
VAE_SHA256 = "a59d7ea697f2942d22002dc3469e8c53db807a6b78f7f5ec03bd4c1f70f98efe"
MODELS = (
    ("scenes", "ft_places2", "6525afb888e55f9b5c74fa0a5d19ca0762d720d6c716fb0f8422fbeb6868a09a"),
    ("faces", "ft_celebahq", "9d26630d91cf9748ed43541669197cf5d63fbb28793d0c1c5dc7c07028ad9045"),
)

USER_AGENT = "moebius-cpu-installer"
ATTEMPTS = 30
RETRY_SECONDS = 5
TIMEOUT_SECONDS = 60
CHUNK = 1 << 20
FREE_DISK_NEEDED = 3 * 10**9


class InstallError(Exception):
    """Something the user can act on. main() prints it without a traceback."""


def progress(label, done, total):
    if total:
        print(f"\r  {label}: {done * 100 // total}% of {total // 10**6} MB ", end="", flush=True)


def transfer(url, part, label):
    """Downloads into `part`, continuing what is there. True once the file is complete."""
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)
    except urllib.error.HTTPError as error:
        if error.code == 416 and have:  # nothing left to send
            return True
        raise
    with response:
        if response.status == 206:
            total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
        else:  # the server ignored the range, so this is the whole file again
            have = 0
            total = int(response.headers.get("Content-Length") or 0)
        with open(part, "ab" if have else "wb") as out:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                have += len(chunk)
                progress(label, have, total)
    return not total or have == total


def fetch(url, path, label, attempts=ATTEMPTS, wait=RETRY_SECONDS):
    """Saves `url` to `path`. Large downloads often drop, so it keeps resuming a .part file."""
    if os.path.exists(path):
        return
    part = path + ".part"
    for attempt in range(1, attempts + 1):
        try:
            if transfer(url, part, label):
                os.replace(part, path)
                print()
                return
            reason = "the connection closed early"
        except urllib.error.HTTPError as error:
            if error.code < 500 and error.code != 429:
                raise InstallError(f"{label}: the server answered {error.code}") from error
            reason = f"the server answered {error.code}"
        except OSError as error:  # URLError, timeouts and resets are all OSError
            if isinstance(error, PermissionError) or error.errno == 28:
                raise InstallError(f"Could not write {part}: {error}") from error
            reason = str(error)
        except Exception as error:  # http.client errors from a dropped connection
            reason = str(error) or type(error).__name__
        print(f"\n  {reason}. Resuming in {wait} s, attempt {attempt} of {attempts}...")
        time.sleep(wait)
    raise InstallError(f"{label}: the download kept failing. Check your connection, then run this again.")


def find_programs():
    """The folder with the compiled programs: next to this file, or built from a source checkout."""
    if all(os.path.isfile(os.path.join(HERE, name)) for name in ("moebius-cli.exe", "moebius-convert.exe")):
        return HERE
    manifest = os.path.join(HERE, "..", "Cargo.toml")
    if not os.path.isfile(manifest) or not shutil.which("cargo"):
        raise InstallError(
            "moebius-cli.exe is missing. Download the release zip instead of the source code:\n"
            "https://github.com/CloudyTabzy/moebius-cpu/releases"
        )
    print("Building the programs, this takes a few minutes...")
    subprocess.run(
        ["cargo", "build", "--release", "--features", "cli", "--bins", "--manifest-path", manifest], check=True
    )
    return os.path.join(HERE, "..", "target", "release")


def copy_plugin(programs, dest):
    os.makedirs(os.path.join(dest, "models"), exist_ok=True)
    try:
        shutil.copy2(os.path.join(HERE, "moebius-inpaint.py"), dest)
        shutil.copy2(os.path.join(programs, "moebius-cli.exe"), dest)
    except PermissionError as error:
        raise InstallError(f"Could not write to {dest}. If GIMP is running, close it and try again.") from error


def copy_models(weights, dest):
    for folder, upstream, _ in MODELS:
        out = os.path.join(dest, "models", folder)
        os.makedirs(out, exist_ok=True)
        for name in ("unet.safetensors", "vae.safetensors"):
            source = os.path.join(weights, upstream, name)
            if not os.path.isfile(source):
                raise InstallError(f"Missing {source}")
            shutil.copy2(source, out)


def installed(dest, folder):
    return all(os.path.isfile(os.path.join(dest, "models", folder, n)) for n in ("unet.safetensors", "vae.safetensors"))


def download_models(programs, dest):
    missing = [model for model in MODELS if not installed(dest, model[0])]
    for folder, _, _ in MODELS:
        if installed(dest, folder):
            print(f"The {folder} model is already installed.")
    if not missing:
        return
    cache = os.path.join(dest, "download")
    vae = os.path.join(cache, "vae")
    os.makedirs(vae, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if free < FREE_DISK_NEEDED:
        raise InstallError(f"This needs about 3 GB of free disk, and only {free / 10**9:.1f} GB is free.")

    fetch(f"{VAE_URL}/config.json", os.path.join(vae, "config.json"), "VAE config")
    vae_file = os.path.join(vae, "diffusion_pytorch_model.bin")
    for folder, upstream, sha256 in missing:
        print(f"Downloading the {folder} model...")
        fetch(f"{VAE_URL}/diffusion_pytorch_model.bin", vae_file, "VAE")
        unet = os.path.join(cache, f"{upstream}.bin")
        fetch(f"{MOEBIUS_URL}/{upstream}/diffusion_pytorch_model.bin", unet, folder)
        print(f"Converting the {folder} model...")
        result = subprocess.run([
            os.path.join(programs, "moebius-convert.exe"),
            "--unet", unet, "--unet-sha256", sha256,
            "--vae", vae, "--vae-sha256", VAE_SHA256,
            "--out", os.path.join(dest, "models", folder),
        ])
        if result.returncode != 0:
            for path in (unet, vae_file):
                if os.path.exists(path):
                    os.remove(path)
            raise InstallError("The downloaded files failed their check, so they were deleted. Run this again.")
        os.remove(unet)
    shutil.rmtree(cache, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Installs Moebius Inpaint for GIMP 3.2.")
    parser.add_argument("--weights", help="folder with ft_places2/ and ft_celebahq/ already converted")
    parser.add_argument("--dest", help="install here instead of GIMP's plug-ins folder")
    args = parser.parse_args(argv)
    try:
        dest = args.dest
        if not dest:
            profile = os.path.join(os.environ.get("APPDATA", ""), "GIMP", "3.2")
            if not os.path.isdir(profile):
                raise InstallError(
                    "GIMP 3.2 was not found for this user.\n"
                    "Install it from https://www.gimp.org/downloads/ and open it once, then run this again."
                )
            dest = os.path.join(profile, "plug-ins", "moebius-inpaint")
        programs = find_programs()
        copy_plugin(programs, dest)
        if args.weights:
            copy_models(args.weights, dest)
        else:
            download_models(programs, dest)
    except InstallError as error:
        print(f"\n{error}\n\nThe installation did not finish.")
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Run this again to carry on.")
        return 1
    print(f"\nInstalled to {dest}")
    print("Restart GIMP, select what to remove, then use Filters > Enhance > Moebius Inpaint...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
