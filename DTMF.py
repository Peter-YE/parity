'''
Frequency Modulation (FM) synthesis module.
To preprocess the low frequency input signal, increase the overall signal frequency and reduce the simulation time.
'''

import numpy as np
from parameters import *

def signal_gen(frequency: float, duration: float) -> np.ndarray:
    """
    Generate a sine wave signal.

    Parameters:
        frequency (float): Frequency of the sine wave in Hz.
        duration (float): Duration of the signal in seconds.
        sample_rate (int): Sample rate in Hz.

    Returns:
        np.ndarray: Generated sine wave signal.
    """
    t = np.arange(0, duration, 1 / sample_rate_audio)  # Time vector
    signal = np.sin(2 * np.pi * frequency * t)
    return signal

def DTMF_gen(number: int, duration: float) -> np.ndarray:
    """
    Generate a DTMF (Dual-Tone Multi-Frequency) signal for a given number.

    Parameters:
        number (str): The DTMF number to generate (e.g., '1234567890').

    Returns:
        np.ndarray: The generated DTMF signal.
    """
    dtmf_frequencies = {
        1: (1209, 697), 2: (1336, 697), 3: (1477, 697),
        4: (1209, 770), 5: (1336, 770), 6: (1477, 770),
        7: (1209, 852), 8: (1336, 852), 9: (1477, 852),
        0: (1336, 941)
    }
    signal1 = signal_gen(dtmf_frequencies[number][0], duration)
    signal2 = signal_gen(dtmf_frequencies[number][1], duration)
    dtmf_signal = signal1 + signal2  # Combine the two frequencies

    return dtmf_signal




def fm_synthesis(input_signal: np.ndarray, modulation_index: float, frequency: float, sample_rate: int) -> np.ndarray:
    """
    Perform frequency modulation synthesis on the input signal.

    Parameters:
        input_signal (np.ndarray): The low frequency input signal to be modulated.
        modulation_index (float): The index of modulation, controlling the depth of modulation.
        frequency (float): The carrier frequency for modulation.
        sample_rate (int): The sample rate of the input signal.

    Returns:
        np.ndarray: The frequency modulated output signal.
    """
    t = np.arange(len(input_signal)) / sample_rate  # Time vector
    modulated_signal = np.sin(2 * np.pi * frequency * t + modulation_index * input_signal)
    return modulated_signal