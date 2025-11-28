# Distance-Based Speech Recognition

This project implements a simple **distance-based speech recognition system** using Python. The system classifies short speech commands by comparing spectral features of the audio signals.

**Dataset:** [Mini Speech Commands](http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip)

## Project Steps

1. **Frame the audio signal**
   The audio waveform is divided into short overlapping frames to capture local temporal information.

2. **Compute spectral features using FFT**
   For each frame, the Fast Fourier Transform (FFT) is applied to extract spectral features that represent the frequency content of the signal.

3. **Compute class templates (training set)**
   For each speech class, the average of all frame-level spectra is computed. This serves as a reference template for the class.

4. **Extract features for testing**
   The test audio signal is framed, and FFT features are computed for each frame.

5. **Distance-based classification**
   The distance between the test features and each class template is computed. The class with the smallest distance is selected as the predicted label.

6. **Evaluate accuracy**
   The predicted classes are compared with the true labels to compute the classification accuracy.

## Notes

* This approach is simple yet effective for small speech command datasets.
* It can serve as a baseline before applying more complex models like CNNs or RNNs.
