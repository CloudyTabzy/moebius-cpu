# Manual install

These steps do by hand what `install.bat` does. You need GIMP 3.2 on Windows, opened at least once,
and about 3 GB of free disk.

## 1. Get the programs

Download `moebius-inpaint-gimp-windows.zip` from
[Releases](https://github.com/CloudyTabzy/moebius-cpu/releases) and unzip it.

## 2. Create the plugin folder

1. Paste this into the address bar of File Explorer and press Enter:

   ```
   %APPDATA%\GIMP\3.2\plug-ins
   ```

   If you're not sure where it is, GIMP shows it under **Edit > Preferences > Folders > Plug-ins**.
2. Create a folder here named exactly `moebius-inpaint`. GIMP only loads the plugin if the folder has
   the same name as the script.
3. Copy `moebius-inpaint.py` and `moebius-cli.exe` from the unzipped folder into it.

## 3. Download the models

Create the folder `%USERPROFILE%\Downloads\moebius`, with a folder named `vae` inside it. Then save
these four files. The three `.bin` files all download as `diffusion_pytorch_model.bin`, so give each
one the name below.

| Link | Save as | Size |
|---|---|---|
| [Scenes model](https://huggingface.co/hustvl/Moebius/resolve/cd01f47fb648219d3fa605806c5ce00b713faa5e/ft_places2/diffusion_pytorch_model.bin?download=true) | `moebius\scenes.bin` | 905 MB |
| [Faces model](https://huggingface.co/hustvl/Moebius/resolve/cd01f47fb648219d3fa605806c5ce00b713faa5e/ft_celebahq/diffusion_pytorch_model.bin?download=true) | `moebius\faces.bin` | 905 MB |
| [VAE](https://huggingface.co/hustvl/PixelHacker/resolve/012fd343158936a265b8a0ee38a791a7a2841f45/vae/diffusion_pytorch_model.bin?download=true) | `moebius\vae\diffusion_pytorch_model.bin` | 167 MB |
| [VAE config](https://huggingface.co/hustvl/PixelHacker/resolve/012fd343158936a265b8a0ee38a791a7a2841f45/vae/config.json?download=true) | `moebius\vae\config.json` | 1 KB |

Your browser may open the VAE config as text instead of downloading it. If so, right-click the link and
choose **Save link as**.

## 4. Convert the models

GIMP can't use these files directly. Open the unzipped folder in File Explorer, type `cmd` in the
address bar and press Enter. Then paste these two commands into the window:

```bat
moebius-convert --unet "%USERPROFILE%\Downloads\moebius\scenes.bin" --unet-sha256 6525afb888e55f9b5c74fa0a5d19ca0762d720d6c716fb0f8422fbeb6868a09a --vae "%USERPROFILE%\Downloads\moebius\vae" --vae-sha256 a59d7ea697f2942d22002dc3469e8c53db807a6b78f7f5ec03bd4c1f70f98efe --out "%APPDATA%\GIMP\3.2\plug-ins\moebius-inpaint\models\scenes"
moebius-convert --unet "%USERPROFILE%\Downloads\moebius\faces.bin" --unet-sha256 9d26630d91cf9748ed43541669197cf5d63fbb28793d0c1c5dc7c07028ad9045 --vae "%USERPROFILE%\Downloads\moebius\vae" --vae-sha256 a59d7ea697f2942d22002dc3469e8c53db807a6b78f7f5ec03bd4c1f70f98efe --out "%APPDATA%\GIMP\3.2\plug-ins\moebius-inpaint\models\faces"
```

Each one takes a few seconds and prints two `wrote` lines. `wrong SHA-256` means that file is damaged or
is the wrong one, so download it again.

## 5. Check the result

The plugin folder should now look like this:

```
%APPDATA%\GIMP\3.2\plug-ins\moebius-inpaint\
    moebius-inpaint.py
    moebius-cli.exe
    models\
        scenes\unet.safetensors
        scenes\vae.safetensors
        faces\unet.safetensors
        faces\vae.safetensors
```

Restart GIMP. The plugin is under **Filters > Enhance > Moebius Inpaint...** You can now delete the
`moebius` folder from Downloads.

## Uninstall

Close GIMP and delete `%APPDATA%\GIMP\3.2\plug-ins\moebius-inpaint`.
