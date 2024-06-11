import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft
import math
import librosa
import pyaudio
import wave


def fft_custom(sig, Fs, zero_padding=False, window=True):
    if zero_padding:
        fft_num = np.power(2, math.ceil(np.log2(len(sig))))
        if window:
            mag = np.fft.fft(sig * np.hanning(len(sig)), fft_num) * 2 / fft_num
            f = np.arange(fft_num) * (Fs / fft_num)
        else:
            mag = np.fft.fft(sig, fft_num) * 2 / fft_num
            f = np.arange(fft_num) * (Fs / fft_num)
    else:
        fft_num = len(sig)
        if window:
            mag = np.fft.fft(sig * np.hanning(len(sig)), fft_num) * 2 / fft_num
            f = np.arange(fft_num) * (Fs / fft_num)
        else:
            mag = np.fft.fft(sig, fft_num) * 2 / fft_num
            f = np.arange(fft_num) * (Fs / fft_num)
    mag[0] /= 2
    return f[:int(fft_num / 2)], abs(mag[:int(fft_num / 2)])


def get_octave_band_base_2(start, multi, oc):
    bands = list()
    for i in range(0, multi * 3 - 2 if oc else multi):
        central_frequency = start * np.power(2, i / 3 if oc else i)
        bands.append(
            [central_frequency, central_frequency / np.power(2, 1 / 6), central_frequency * np.power(2, 1 / 6)])
    return np.asarray(bands)


def record_audio(file_path, duration, sr):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = sr
    RECORD_SECONDS = duration

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    print("Recording...")

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def a_weighting(frequencies):
    frequencies[0] = frequencies[0] + 1e-12
    ra = (12194 ** 2) * (frequencies ** 4)
    denom = (frequencies ** 2 + 20.6 ** 2) * (frequencies ** 2 + 12194 ** 2) * \
            np.sqrt((frequencies ** 2 + 107.7 ** 2) * (frequencies ** 2 + 737.9 ** 2))
    a_weight = ra / denom
    a_weight_db = 20 * np.log10(a_weight) + 2.00
    return a_weight_db


def measure_spl(file_path, record_time=0.5, sr=44100, start_f=31.25, multi_f=8, oc_f=0, gain=0, A_weighting=False):
    # record_audio(file_path, duration=record_time, sr=sr)
    # print("Audio recorded and saved as 'recorded_audio.wav'.")

    sig, Fs = librosa.load(file_path)
    print(sig)

    sig = sig * 10 ** (gain / 20)

    N = len(sig)
    n = list(np.arange(N))
    t = np.asarray([temp / Fs for temp in n])

    temp = [temp * temp for temp in sig]
    p_square_time = np.sum(temp) / len(temp)
    print("p_square_time: ", p_square_time)

    f, mag_fft = fft_custom(sig, Fs)

    if A_weighting:
        a_weight = a_weighting(f)
        mag_fft = mag_fft * 10**(a_weight / 20)

    temp = [(temp / np.sqrt(2)) * (temp / np.sqrt(2)) for temp in mag_fft]
    p_square_frequency = np.sum(temp)
    print("p_square_frequency: ", p_square_frequency)

    spl_overall = 20 * np.log10(p_square_time / 0.0000000004)
    print("声压级(DB):", spl_overall)

    bands = get_octave_band_base_2(start=start_f, multi=multi_f, oc=oc_f)
    spl_of_bands = list()
    f = list(f)
    index_start = 0
    for band in bands:
        index_stop = np.where(f < band[2])[0][-1]
        band_frequencies_mag = mag_fft[index_start:index_stop]
        temp = [temp ** 2 / 2 for temp in band_frequencies_mag]
        p_square_frequency_band = np.sum(temp)
        spl_ = 20 * np.log10(p_square_frequency_band / 0.0000000004)
        if band[0] <= Fs / 2:
            spl_of_bands.append([band[0], spl_])
        else:
            break
        index_start = index_stop

    plt.clf()
    spl_values = list()
    xticks = list()

    for temp in spl_of_bands:
        spl_values.append(temp[1])
        if temp[0] % start_f > 0.0:
            xticks.append(' ')
        elif temp[0] >= 1000:
            xticks.append(str(int(temp[0] / 1000)) + 'k')
        else:
            xticks.append(str(int(temp[0])))

    plt.figure(2)
    plt.bar(range(len(spl_values)), spl_values, facecolor='b', width=0.8, zorder=3)
    plt.title("1/3 octave SPL || Overall SPL is %f " % np.round(spl_overall, 3))
    plt.xticks(list(range(len(spl_values))), xticks, rotation=30)
    plt.grid(axis='x', linestyle='--', zorder=0)
    plt.xlabel('Freq/Hz')
    plt.ylabel('SPL')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    record_time = 0.5  # 录制音频长度
    sr = 44100  # 采样率
    start_f = 31.25  # 起始频率
    multi_f = 8  # 频带数
    oc_f = 0  # 0是倍频程，1是1/3倍频程
    gain = 10  # 增益大小
    A_weighting = True  # 是否使用A集权
    measure_spl('recorded_audio.wav', record_time, sr, start_f, multi_f, oc_f, gain, A_weighting)
