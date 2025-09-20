import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import hashlib
from settings import *


def audio_to_spectrogram(file_path, sr=SAMPLE_RATE, n_fft=FFT_SIZE, hop_length=HOP_LENGTH, mels=True, show=False):
    """
    преобразует аудиофайл в спектрограмму или мел-спектрограмму.

    :param file_path: путь к аудиофайлу (mp3, wav, flac и т.д.)
    :param sr: частота дискретизации (по умолчанию 22050 Гц)
    :param n_fft: размер окна для преобразования Фурье
    :param hop_length: шаг окна
    :param mels: если True, строится мел-спектрограмма
    :param show: если True, отображает спектрограмму с помощью matplotlib
    :return: np.ndarray спектрограмма (в dB)
    """
    # Загружаем аудио
    y, sr = librosa.load(file_path, sr=sr)

    if mels:
        # Мел-спектрограмма
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
    else:
        # STFT спектрограмма
        S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

    if show:
        plt.figure(figsize=(10, 4))
        if mels:
            librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                                     x_axis="time", y_axis="mel")
            plt.title("Mel-Spectrogram")
        else:
            librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                                     x_axis="time", y_axis="log")
            plt.title("Spectrogram")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.show()
    return S_db

def find_peaks(Sp):
    Sp_max = maximum_filter(Sp, size=PEAK_BOX_SIZE, mode='constant', cval=0.0)
    peak_goodmask = Sp_max == Sp
    y_peaks, x_peaks = peak_goodmask.nonzero()
    peak_values = Sp[y_peaks, x_peaks]
    i = peak_values.argsort()[::-1]
    # get co-ordinates into arr
    j = [(y_peaks[idx], x_peaks[idx]) for idx in i]
    total = Sp.shape[0] * Sp.shape[1]
    # in a square with a perfectly spaced grid, we could fit area / PEAK_BOX_SIZE^2 points
    # use point efficiency to reduce this, since it won't be perfectly spaced
    # accuracy vs speed tradeoff
    peak_target = int((total / (PEAK_BOX_SIZE ** 2)) * POINT_EFFICIENCY)
    return j[:peak_target]

def generate_hashes(peaks, fan_value=15):
    """
    peaks: list of (freq_bin, time_bin) sorted by time (or by energy)
    fan_value: сколько целевых точек брать после опорной
    Возвращает список троек (hash, t1)
    """
    # сортируем по времени
    peaks_sorted = sorted(peaks, key=lambda x: x[1])
    hashes = []
    for i in range(len(peaks_sorted)):
        freq1, t1 = peaks_sorted[i]
        # для каждой опорной точки берем следующие fan_value точек в пределах окна
        for j in range(1, fan_value+1):
            if i + j >= len(peaks_sorted):
                break
            freq2, t2 = peaks_sorted[i+j]
            dt = t2 - t1
            if dt < MIN_HASH_TIME_DELTA or dt > MAX_HASH_TIME_DELTA:
                continue
            # формируем текстовый ключ и хешируем
            key = f"{freq1}|{freq2}|{dt}"
            h = hashlib.sha1(key.encode('utf-8')).hexdigest()[:20]
            hashes.append((h, t1))
    return hashes

