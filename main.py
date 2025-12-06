import numpy as np
import os
from scipy.io import wavfile
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import shutil
import seaborn as sns
import matplotlib.pyplot as plt

def frame_signal(signal, frame_length, frame_step):
    signal_length = len(signal)
    num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1
    pad_length = (num_frames - 1) * frame_step + frame_length
    padded_signal = np.pad(signal, (0, pad_length - signal_length), mode='constant')
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * frame_step
        frames[i] = padded_signal[start:start + frame_length]
    return frames

def compute_fft_features(frames, use_log):
    window = np.hamming(frames.shape[1])
    windowed_frames = frames * window
    fft_frames = np.abs(fft(windowed_frames, axis=1))
    fft_features = fft_frames[:, :fft_frames.shape[1]//2]
    if use_log:
        fft_features = np.log(fft_features + 1e-10)
    return fft_features

def extract_features(audio_path, frame_length, frame_step, use_log):
    sr, signal = wavfile.read(audio_path)
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
    signal = signal.astype(float)
    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    frames = frame_signal(signal, frame_length, frame_step)
    return compute_fft_features(frames, use_log)

def compute_class_templates(data_path, classes, frame_length, frame_step, use_log):
    templates = {}
    for class_name in classes:
        class_path = os.path.join(data_path, class_name)
        audio_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        all_features = []
        for audio_file in audio_files:
            audio_path = os.path.join(class_path, audio_file)
            try:
                features = extract_features(audio_path, frame_length, frame_step, use_log)
                all_features.append(features)
            except:
                pass
        max_frames = max(f.shape[0] for f in all_features)
        padded = []
        for f in all_features:
            if f.shape[0] < max_frames:
                pad = ((0, max_frames - f.shape[0]), (0, 0))
                f = np.pad(f, pad, mode='constant')
            padded.append(f)
        templates[class_name] = np.mean(padded, axis=0)
    return templates

def compute_distance(test_features, template, distance_type):
    max_frames = max(test_features.shape[0], template.shape[0])
    if test_features.shape[0] < max_frames:
        pad = ((0, max_frames - test_features.shape[0]), (0, 0))
        test_features = np.pad(test_features, pad, mode='constant')
    if template.shape[0] < max_frames:
        pad = ((0, max_frames - template.shape[0]), (0, 0))
        template = np.pad(template, pad, mode='constant')
    if distance_type == 'euclidean':
        return np.sqrt(np.sum((test_features - template) ** 2))
    test_flat = test_features.flatten()
    temp_flat = template.flatten()
    if distance_type == 'cosine':
        dot = np.dot(test_flat, temp_flat)
        n1 = np.linalg.norm(test_flat)
        n2 = np.linalg.norm(temp_flat)
        return 1 - (dot / (n1 * n2 + 1e-10))
    corr = np.corrcoef(test_flat, temp_flat)[0, 1]
    return 1 - corr

def predict_class(test_features, templates, distance_type):
    distances = {c: compute_distance(test_features, t, distance_type) for c, t in templates.items()}
    return min(distances, key=distances.get), distances

def evaluate_model(data_path, templates, test_files, frame_length, frame_step, use_log, distance_type):
    correct = 0
    total = len(test_files)
    predictions = []
    actuals = []
    for actual_class, file_name in test_files:
        audio_path = os.path.join(data_path, actual_class, file_name)
        try:
            features = extract_features(audio_path, frame_length, frame_step, use_log)
            pred, _ = predict_class(features, templates, distance_type)
            predictions.append(pred)
            actuals.append(actual_class)
            if pred == actual_class:
                correct += 1
        except:
            total -= 1
    return correct / total if total > 0 else 0, predictions, actuals

def main():
    data_path = '/home/yahia/DSP/mini_speech_commands/mini_speech_commands'
    frame_length = 512
    frame_step = 256
    train_ratio = 0.8
    use_log = False
    distance_type = 'euclidean'
    if not os.path.exists(data_path):
        print("Dataset not found")
        return
    classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and not d.startswith('__')]
    all_files = []
    for c in classes:
        audio_files = [f for f in os.listdir(os.path.join(data_path, c)) if f.endswith('.wav')]
        all_files.extend([(c, f) for f in audio_files])
    train_files, test_files = train_test_split(all_files, train_size=train_ratio, random_state=42, stratify=[a for a, _ in all_files])
    train_path = 'data/train_speech_commands'
    os.makedirs(train_path, exist_ok=True)
    for c in classes:
        os.makedirs(os.path.join(train_path, c), exist_ok=True)
    for c, f in train_files:
        src = os.path.join(data_path, c, f)
        dst = os.path.join(train_path, c, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    templates = compute_class_templates(train_path, classes, frame_length, frame_step, use_log)
    accuracy, predictions, actuals = evaluate_model(data_path, templates, test_files, frame_length, frame_step, use_log, distance_type)
    print(f"ACCURACY: {accuracy * 100:.2f}%")
    for c in classes:
        correct = sum(1 for p, a in zip(predictions, actuals) if a == c and p == c)
        total = sum(1 for a in actuals if a == c)
        print(f"{c}: {(correct/total*100 if total>0 else 0):.2f}% ({correct}/{total})")

    class_indices = {c: i for i, c in enumerate(classes)}
    y_true = [class_indices[a] for a in actuals]
    y_pred = [class_indices[p] for p in predictions]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()
