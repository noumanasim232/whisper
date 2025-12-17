# Running on Ubuntu

Yes, you can run this on Ubuntu! You just need to install a few system dependencies first.

## 1. Install System Dependencies

Open your terminal and run:

```bash
sudo apt update
sudo apt install ffmpeg libportaudio2 xclip python3-tk python3-dev
```

*   `ffmpeg`: Required by Whisper for audio processing.
*   `libportaudio2`: Required by `sounddevice` for microphone access.
*   `xclip`: Required by `pyperclip` to copy text to the clipboard.
*   `python3-tk` / `python3-dev`: Required by `pyautogui`.

## 2. Install Python Libraries

If you haven't already, install the Python requirements:

```bash
pip install openai-whisper sounddevice numpy pyautogui pyperclip
```

## 3. Important Note for Ubuntu 22.04+ (Wayland)

Ubuntu 22.04 and later use **Wayland** by default instead of X11. `pyautogui` (the tool that types the text) **does not work well on Wayland**.

**Solution:**
1.  Log out of Ubuntu.
2.  Click your username.
3.  Click the gear icon (⚙️) in the bottom right.
4.  Select **"Ubuntu on Xorg"**.
5.  Log in.

Now the script should work perfectly!
