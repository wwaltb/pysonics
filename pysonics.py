"""
Pysonics: A python mmodule for real-time audio analysis and visualization.
"""

import pyaudio
import numpy as np
import pygame

### Audio Capture ###

def setup_audio_stream(sample_rate: int, window_size: int, device_name="default"):
    """
    Sets up and returns a PyAudio instance and stream.
    :param sample_rate: Rate to sample the stream at (Hz)
    :param window_size: Number of samples per frame
    :raises ValueError: If the specified device name is not found
    :return: PyAudio instance and the opened PyAudio stream
    """
    p = pyaudio.PyAudio()

    # Find the correct device index
    device_index = None
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        dev_name = str(dev_info.get("name", "")).lower()
        if device_name.lower() in dev_name:
            device_index = i
            break
    
    # Print vailable devices and raise error if the specified device is not
    # found
    if device_index is None:
        available_devices = "\n".join(
            [str(p.get_device_info_by_index(i).get("name", "Unknown")) for i in
                range(p.get_device_count())]
        )
        raise ValueError(f"Could not find device matching '{device_name}'. \
            Available devices:\n{available_devices}")

    stream: pyaudio.Stream = p.open(format=pyaudio.paInt16, channels=1,
                                    rate=sample_rate, input=True,
                                    frames_per_buffer=window_size)

    return p, stream

def capture_time_domain_signal(stream, window_size):
    """
    Captures a single chunk of audio data using an open PyAudio stream and
    returns it as a NumPy array.
    :param stream: Open PyAudio stream
    :param window_size: Number of samples per frame
    :return: 1D array of recorded audio samples
    """
    return np.frombuffer(stream.read(window_size, exception_on_overflow=False),
                         dtype=np.int16)

def close_audio_stream(p, stream):
    """
    Closes the PyAudio stream and terminates the PyAudio instance.
    :param p: PyAudio instance
    :param stream: Open PyAudio stream
    """
    stream.stop_stream()
    stream.close()
    p.terminate()

### Audio Analysis ###

def remove_dc_offset(data: np.ndarray):
    return data - np.mean(data)

def apply_hamming_window(data: np.ndarray):
    return data * np.hamming(len(data))

def compute_fft(data: np.ndarray):
    """
    Computes the FFT of some data.
    :param data: Input signal (1D array)
    :return: FFT magnitudes
    """
    data = remove_dc_offset(data)
    data = apply_hamming_window(data)
    window_size = int(len(data) * 0.5)
    fft_result = np.fft.fft(data)
    magnitudes = np.sqrt(fft_result.real[:window_size] ** 2 + fft_result.imag[:window_size] ** 2)
    return magnitudes[:window_size]

def smooth_data(data: np.ndarray, window_size: int):
    """
    Computes a moving average of the data based on a smoothing window size.
    :param data: Input data (1D array)
    :param window_size: Number of samples around the current sample to use for
    averaging
    :return: Smoothed data
    """
    output = np.copy(data)
    data_length = len(data)

    for i in range(data_length):
        start_idx = max(0, i - window_size)
        end_idx = min(data_length - 1, i + window_size)
        output[i] = np.mean(data[start_idx:end_idx + 1])
    
    return output

smoothed_data: np.ndarray | None = None
def smooth_data_over_time(data: np.ndarray, smoothing_factor: float = 0.05):
    global smoothed_data
    if smoothed_data is None:
        smoothed_data = data

    for i in range(len(data)):
        smoothed_val = smoothing_factor * data[i] + (1 - smoothing_factor) * smoothed_data[i]
        smoothed_data[i] = smoothed_val
    return smoothed_data

def remove_noise_cfar(data: np.ndarray, size: int, gap: int, bias: float):
    """
    Removes peaks from the frequency domain using a noise gate.
    :param data: Input data (1D array)
    :param size: Number of samples around the current sample for computing the noise gate
    :param gap: Number of samples to exclude around the current sample
    :param bias: Noise gate multiplier
    :return: Trimmed data
    """
    filtered_data = np.copy(data)
    data_length = len(data)
    noise_gate = []

    for i in range(data_length):
        sample_sum = []
        start_idx = max(0, i - size)
        end_idx = min(data_length - 1, i + size)
        gap_start_idx = max(0, i - gap)
        gap_end_idx = min(data_length - 1, i + gap)

        sample_sum.extend(data[start_idx:gap_start_idx])
        sample_sum.extend(data[gap_end_idx + 1:end_idx + 1])

        noise_gate.append(np.mean(sample_sum) * bias)

    for i in range(data_length):
        filtered_data[i] = data[i] if data[i] > noise_gate[i] else 0
    
    return filtered_data

# https://www.youtube.com/watch?v=2s3aJfRr9gE
def compute_entropy(data: np.ndarray):
    entropy = 0.0
    for i in range(len(data)):
        # p is the spectral density of the bin, or the percent of the total
        # amplitude it containts. For calculating entropy, it is helpful to
        # think of this as the probability of the bin containing energy.
        p = data[i] / (sum(data) + 1e-10)
        # Calculate the entropy addition of each bin:
        #
        # -log(p) is the 'self-information' of the bin. It is a measure of how
        # surprising it is to see energy in that bin. If p is close to 1, then
        # it is not surprising at all to see energy there, but if it is small
        # it this value will be greater, meaning it is very suprising to see
        # energy there. This makes sense as p was calculated as the percent of
        # the total energy that the bin contains.
        #
        # We weight this 'self-information' value with the actual percent of
        # the total amplitude that bin has so that bins with very little energy
        # do not dominate the entropy value.
        entropy += p * -(np.log2(p))

    # normalize the entropy value by the log2 of the number of bins, which is
    # the maximum possible entropy value.
    return entropy / np.log2(len(data))

# A measure of how flat the spectrum is. Computes the ratio of geometric mean
# to arithmetic mean. The geometric mean gets further way from the arithmetic
# mean as the spectrum becomes more spiky. This means a perfectly even
# distribution has a flatness of 1, and a single spike has a flatness close to
# 0.
def compute_flatness(data: np.ndarray):
    arithmetic_mean = np.mean(data)
    # calculate the geometric mean (needs more precision to calculate the
    # product of all the amplitude values)
    geometric_mean: np.float128 = np.float128(1.)
    for i in range(len(data)):
        geometric_mean *= data[i]
    geometric_mean = geometric_mean ** np.float128(1. / len(data))
    return geometric_mean / arithmetic_mean

        
# https://en.wikipedia.org/wiki/Spectral_flux
# Computes the spectral flux between two frames. Gives a measure of how much
# the spectral energy has increased between frames, normalized by the total
# energy. This value can be used to detect onsets.
def compute_flux(currentData: np.ndarray, previousData: np.ndarray):
    rectified_flux = 0.0
    for i in range(len(currentData)):
        rectified_flux += pow(max(0, currentData[i] - previousData[i]), 2)
    return rectified_flux / sum(currentData ** 2)

### Audio Visualization ###

norm_max = 10000
def normalize_data(data: np.ndarray, smoothing_factor = 0.05):
    """
    Normalizes an array of data by the max value smoothed by the previously
    recorded maximum (or 10000 if none) using a smoothing factor.  
    :param data: The array of data
    :param smoothing_factor: The factor to smooth the max value by (lower is smoother)
    return: The normalized array
    """
    global norm_max
    data_max = np.max(data)
    if data_max > norm_max:
        norm_max = data_max
    else:
        norm_max = smoothing_factor * data_max + (1 - smoothing_factor) * norm_max
    return data / norm_max

def normalize_data2(data1: np.ndarray, data2: np.ndarray, smoothing_factor = 0.05):
    """
    Normalizes two arrays of data by the max value found in either one. Smooths
    the maximum value by the previously recorded maximum (or 10000 if none)
    using a smoothing factor.  
    :param data1: The first array
    :param data2: The second array
    :param smoothing_factor: The factor to smooth the max value by (lower is smoother)
    return: The two normalized arrays
    """
    global norm_max
    data_max = np.max(data1) if np.max(data1) > np.max(data2) else np.max(data2)
    if data_max > norm_max:
        norm_max = data_max
    else:
        norm_max = smoothing_factor * data_max + (1 - smoothing_factor) * norm_max

    return data1 / norm_max, data2 / norm_max

def setup_window(width=1200, height=800):
    """
    Sets up the Pygame window for real-time FFT visualization.
    :param width: Width of the window
    :param height: Height of the window
    :return: Pygame screen
    """
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("Real-time Frequency Spectrum")
    return screen

def draw_fft_spectrum(screen, data: np.ndarray):
    """
    Draws the real-time FFT spectrum using Pygame.
    :param screen: Pygame screen to draw on
    :param data: FFT data to draw
    """
    screen.fill((31, 31, 40))
    width, height = screen.get_size()
    num_bars = len(data)
    bar_width = width / num_bars

    norm_data = normalize_data(data)
    
    for i, value in enumerate(norm_data):
        bar_height = int(value * height)
        pygame.draw.rect(screen, (118, 148, 106), (i * bar_width, height - bar_height, bar_width - 2, bar_height))
    
    pygame.display.flip()

def draw_reflected_fft_spectrums(screen: pygame.Surface, data_top: np.ndarray, data_bot: np.ndarray):
    """
    Draws two FFT spectrums, one above a middle line and one underneath.
    :param screen: Pygame screen to draw on
    :param data_top: FFT data to draw above
    :param data_bot: FFT data to draw underneath
    """
    screen.fill((31, 31, 40))
    width, height = screen.get_size()
    num_bars = len(data_top)
    bar_width = int(width / num_bars)
    x_offset = (width - bar_width * num_bars) * 0.5

    norm_data_top, norm_data_bot = normalize_data2(data_top, data_bot)

    for i, (val_top, val_bot) in enumerate(zip(norm_data_top, norm_data_bot)):
        bar_height_top = int(val_top * height * 0.45)
        bar_height_bot = int(val_bot * height * 0.45)

        pygame.draw.rect(screen, (118, 148, 106), (x_offset + i * bar_width, height * 0.5 - bar_height_top, bar_width - 1, bar_height_top))
        pygame.draw.rect(screen, (195, 64, 67), (x_offset + i * bar_width, height * 0.5, bar_width - 1, bar_height_bot))
        pygame.draw.rect(screen, (220, 215, 186), (x_offset - bar_width * 2, height * 0.5 - 1, width - x_offset * 2 + bar_width * 4, 2))

    pygame.display.flip()

def close_window():
    """
    Closes the Pygame window.
    """
    pygame.quit()
