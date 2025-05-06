import pygame
import pysonics as ps
import numpy as np

def real_time_plot():
    sample_rate = 8192
    window_size = 256

    # init pysonics audio stream and gui window
    p, stream = ps.setup_audio_stream(sample_rate, window_size)
    screen = ps.setup_window()
    
    fd_previous = np.zeros(window_size >> 1)

    # graphics loop:
    done = False
    while not done:
        # handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        # capture window of audio data
        data = ps.capture_time_domain_signal(stream, window_size)

        # compute fft and analyze
        fd_data = ps.compute_fft(data)
        fd_smoothed = ps.smooth_data(fd_data, 1)
        fd_filtered = ps.remove_noise_cfar(fd_data, 6, 2, 1)
        fd_blended = ps.smooth_data_over_time(fd_filtered, 0.16)
        fd_percussion = fd_data - fd_blended

        # draw filtered and raw spectrum
        # ps.draw_reflected_fft_spectrums(screen, fd_filtered, fd_smoothed)
        ps.draw_reflected_fft_spectrums(screen, fd_blended, fd_percussion)

        energy_threshold = 90000000
        flux_threshold = 0.7
        if sum(fd_data ** 2) > energy_threshold and ps.compute_flux(fd_data, fd_previous) > flux_threshold and ps.compute_entropy(fd_data) > 0.7:
            print("")
            print("Percussion detected!")
            print("entropy: " + str(ps.compute_entropy(fd_data)))
            print("flatness: " + str(ps.compute_flatness(fd_data)))
            print("flux: " + str(ps.compute_flux(fd_data, fd_previous)))
            print("energy: " + str(sum(pow(fd_data, 2))))

        fd_previous = fd_data

    # close pysonics
    ps.close_audio_stream(p, stream)
    ps.close_window()

if __name__ == "__main__":
    real_time_plot()
