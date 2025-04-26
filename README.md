# Pysonics
A python module to streamline real-time audio processing, analysis and visualization.

![Example visualization of raw and processed frequency spectrums](https://github.com/wwaltb/pysonics/blob/main/misc/visualizer.gif)
## Installation/Setup
### Linux
#### Arch Installation:
- Pyaudio: `sudo pacman -S python-pyaudio`
- Pygame: `sudo pacman -S python-pygame`
- (Optional) PulseAudio Volume Control: `sudo pacman -S pavucontrol`

#### Setup Input Device:
- Open `pavucontrol` -> `Input Devices`
- Make sure `show` is set to `All Input Devices` or `Monitors`
- Set the monitor of your desired audio source as the default input device

### MacOS
#### Install via Homebrew and pip:
1. Install Dependencies: 
    - Portaudio: `brew install portaudio`
    - Pyaudio: `pip install pyaudio`
    - Pygame: `pip3 install pygame`

2. Create a Monitor of System Audio:
    - Install BlackHole: `brew install blackhole-2ch`
    - Restart your computer
    - Create a Multi-Output Device following [this
    guide](https://github.com/ExistentialAudio/BlackHole/wiki/Multi-Output-Device)
        - Make sure it uses BlackHole and the output device you want to capture
    - Configure your system's input and output devices:
        - `System Settings ` -> `Sound` -> `Output` -> `Multi-Output Device`
        - `System Settings ` -> `Sound` -> `Input` -> `BlackHole 2ch`
