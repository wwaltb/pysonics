import pygame
import pysonics as ps

def real_time_plot():
    sample_rate = 8192
    window_size = 256

    # init pysonics audio stream and gui window
    p, stream = ps.setup_audio_stream(sample_rate, window_size)
    screen = ps.setup_window()
    
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
        fd_filtered = ps.remove_noise_cfar(fd_smoothed, 3, 1, 1.2)
        fd_blended = ps.smooth_data_over_time(fd_filtered, 0.16)

        # draw filtered and raw spectrum
        ps.draw_reflected_fft_spectrums(screen, fd_blended, fd_filtered)

    # close pysonics
    ps.close_audio_stream(p, stream)
    ps.close_window()

if __name__ == "__main__":
    real_time_plot()
