# Masterplan: Eigene Speech-to-Text (STT) Engine

## Inhaltsverzeichnis

1. [Übersicht & Zielsetzung](#1-übersicht--zielsetzung)
2. [Architektur-Überblick](#2-architektur-überblick)
3. [Kernkomponenten im Detail](#3-kernkomponenten-im-detail)
4. [Technologie-Stack](#4-technologie-stack)
5. [Daten & Datasets](#5-daten--datasets)
6. [Modellarchitekturen (Optionen)](#6-modellarchitekturen-optionen)
7. [Audio-Preprocessing Pipeline](#7-audio-preprocessing-pipeline)
8. [Training Pipeline](#8-training-pipeline)
9. [Decoding & Postprocessing](#9-decoding--postprocessing)
10. [Evaluation & Metriken](#10-evaluation--metriken)
11. [Deployment & API](#11-deployment--api)
12. [Projektstruktur](#12-projektstruktur)
13. [Implementierungs-Phasen](#13-implementierungs-phasen)
14. [Hardware-Anforderungen](#14-hardware-anforderungen)
15. [Bekannte Herausforderungen](#15-bekannte-herausforderungen)
16. [Referenzen & Ressourcen](#16-referenzen--ressourcen)

---

## 1. Übersicht & Zielsetzung

Ziel ist der Aufbau einer **vollständig eigenen, funktionsfähigen Speech-to-Text Engine**, die:

- Gesprochene Sprache (Audio) in geschriebenen Text umwandelt
- End-to-End trainierbar ist (kein manuelles Feature-Engineering nötig)
- Modular aufgebaut ist (Komponenten austauschbar)
- Zumindest auf Englisch und optional auf Deutsch funktioniert
- Lokal lauffähig ist (kein Cloud-Zwang)

### Ansatz: End-to-End Deep Learning

Moderne STT-Systeme verwenden **End-to-End Deep Learning**, d.h. ein einziges neuronales Netz lernt die Abbildung von Audio-Signalen direkt auf Text. Dies ersetzt das klassische Pipeline-Modell (Akustisches Modell + Aussprache-Lexikon + Sprachmodell).

---

## 2. Architektur-Überblick

```
┌─────────────────────────────────────────────────────────────┐
│                    STT Engine Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │  Audio    │──▶│  Feature     │──▶│  Encoder          │   │
│  │  Input    │   │  Extraction  │   │  (CNN + Transform)│   │
│  │ (.wav)    │   │  (Mel-Spec)  │   │                    │   │
│  └──────────┘   └──────────────┘   └────────┬───────────┘   │
│                                              │               │
│                                              ▼               │
│                                    ┌──────────────────┐     │
│                                    │   Decoder         │     │
│                                    │   (CTC / Att.)    │     │
│                                    └────────┬─────────┘     │
│                                              │               │
│                                              ▼               │
│                                    ┌──────────────────┐     │
│                                    │  Postprocessing   │     │
│                                    │  (LM, Punctuation)│     │
│                                    └────────┬─────────┘     │
│                                              │               │
│                                              ▼               │
│                                    ┌──────────────────┐     │
│                                    │   Text Output     │     │
│                                    └──────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Kernkomponenten im Detail

### 3.1 Audio Input & Vorverarbeitung

| Schritt | Beschreibung |
|---------|-------------|
| **Audio laden** | WAV/FLAC/MP3 einlesen, auf Mono konvertieren |
| **Resampling** | Einheitliche Sample Rate: **16.000 Hz** (Standard für STT) |
| **Normalisierung** | Amplitude auf [-1, 1] normalisieren |
| **Voice Activity Detection (VAD)** | Stille-Abschnitte erkennen und optional entfernen |
| **Noise Reduction** | Optional: Hintergrundrauschen reduzieren |

### 3.2 Feature Extraction

Die rohen Audio-Samples werden in kompakte Feature-Vektoren transformiert:

| Feature-Typ | Beschreibung | Empfehlung |
|-------------|-------------|------------|
| **Mel-Spektrogramm** | Frequenzdarstellung mit menschlichem Hörempfinden gewichtet | ✅ Standard für moderne Modelle |
| **MFCC** | Mel-Frequency Cepstral Coefficients – kompakte cepstrale Darstellung | Gut für kleinere Modelle |
| **Raw Waveform** | Direkte Nutzung der Rohdaten (Wav2Vec2-Ansatz) | ✅ Für Self-Supervised Learning |
| **Log-Mel Filterbanks** | Logarithmierte Mel-Filterbank-Energien | ✅ Häufig bei Transformer-Modellen |

**Empfohlene Parameter für Mel-Spektrogramm:**
- Window Size: 25 ms
- Hop Size (Stride): 10 ms
- Anzahl Mel-Bins: 80
- FFT Size: 512 oder 1024
- Frequenzbereich: 20 Hz – 8000 Hz

### 3.3 Encoder (Akustisches Modell)

Der Encoder wandelt Feature-Sequenzen in hochdimensionale Repräsentationen um.

**Optionen:**

| Architektur | Stärken | Schwächen |
|-------------|---------|-----------|
| **Conformer** | Beste Accuracy, kombiniert CNN + Transformer | Rechenintensiv |
| **Transformer** | Parallelisierbar, starke Kontextmodellierung | Braucht viel Daten |
| **CNN (Jasper/QuartzNet)** | Schnell, effizient | Weniger Kontext |
| **RNN/LSTM** | Gut für sequenzielle Daten | Schwer parallelisierbar |
| **Wav2Vec2** | Self-Supervised Pretraining, wenig gelabelte Daten nötig | Hoher Speicherbedarf |

**Empfehlung: Conformer-Encoder** (aktueller State-of-the-Art)

### 3.4 Decoder

Der Decoder wandelt die Encoder-Ausgabe in Text-Tokens um.

| Methode | Beschreibung | Einsatz |
|---------|-------------|---------|
| **CTC (Connectionist Temporal Classification)** | Alignment-freies Training, gibt Zeichenwahrscheinlichkeiten pro Frame aus | Schnell, einfach |
| **Attention-based (LAS)** | Autoregressive Decoder mit Aufmerksamkeitsmechanismus | Genauer, langsamer |
| **CTC + Attention (Hybrid)** | Kombiniert CTC für Alignment mit Attention für Genauigkeit | ✅ Beste Wahl |
| **Transducer (RNN-T)** | Streaming-fähig, für Echtzeit-Anwendungen | Komplex |

**Empfehlung: CTC + Attention Hybrid** (wie bei ESPnet/Whisper)

### 3.5 Language Model (Sprachmodell) – Postprocessing

Ein externes Sprachmodell verbessert die Textqualität erheblich:

- **n-gram LM** (KenLM): Leichtgewichtig, schnell, einfach zu trainieren
- **Transformer LM**: Genauer, aber ressourcenintensiver
- **Shallow Fusion**: LM-Score wird beim Beam Search dazuaddiert
- **Deep Fusion**: LM in den Decoder integriert

---

## 4. Technologie-Stack

### Programmiersprache
- **Python 3.10+** (Hauptsprache)
- **C++** (optional für Performance-kritische Teile)

### Deep Learning Framework
| Framework | Vorteile | Empfehlung |
|-----------|---------|------------|
| **PyTorch** | Flexibel, große Community, torchaudio | ✅ Empfohlen |
| **TensorFlow** | Production-ready, TFLite für Mobile | Alternative |

### Zentrale Bibliotheken

```
# Core
torch>=2.0              # Deep Learning Framework
torchaudio>=2.0         # Audio-Verarbeitung & Features
numpy                   # Numerische Berechnungen

# Audio Processing
librosa                 # Audio-Analyse & Features
soundfile               # Audio I/O
webrtcvad               # Voice Activity Detection
noisereduce             # Rauschunterdrückung

# Modelling
transformers            # Pretrained Modelle (Wav2Vec2, Whisper)
sentencepiece           # Tokenization (BPE/Unigram)
ctcdecode               # CTC Beam Search Decoder
pyctcdecode             # CTC Decoder mit LM-Support
kenlm                   # n-gram Language Model

# Training
accelerate              # Distributed Training
wandb                   # Experiment Tracking
datasets                # HuggingFace Datasets

# Serving
fastapi                 # REST API
uvicorn                 # ASGI Server
onnxruntime             # Model Inference Optimierung
```

---

## 5. Daten & Datasets

### 5.1 Öffentliche Datasets

| Dataset | Sprache | Stunden | Beschreibung |
|---------|---------|---------|-------------|
| **LibriSpeech** | EN | 960h | Standard-Benchmark, saubere Audiobooks |
| **Common Voice** (Mozilla) | Multi | 20.000h+ | Crowdsourced, viele Sprachen inkl. DE |
| **GigaSpeech** | EN | 10.000h | Große Vielfalt (YouTube, Podcasts) |
| **VoxPopuli** | Multi | 400.000h+ | EU-Parlament Aufnahmen |
| **MLS (Multilingual LibriSpeech)** | Multi | 50.000h+ | Hörbücher in 8 Sprachen |
| **Fleurs** | Multi | 12h/Sprache | 102 Sprachen, gutes Eval-Set |
| **Tuda-DE** | DE | 184h | Deutsch, Spontansprache |

### 5.2 Datenformat

Jeder Datenpunkt besteht aus:
```json
{
  "audio_path": "data/audio/sample_001.wav",
  "text": "das ist ein beispieltext",
  "duration": 4.2,
  "sampling_rate": 16000,
  "speaker_id": "speaker_042",
  "language": "de"
}
```

### 5.3 Datenaugmentierung

Kritisch für robuste Modelle:

| Technik | Beschreibung |
|---------|-------------|
| **SpecAugment** | Maskierung von Zeit- und Frequenzbändern im Spektrogramm |
| **Speed Perturbation** | Geschwindigkeit ×0.9, ×1.0, ×1.1 |
| **Noise Injection** | Hintergrundgeräusche hinzufügen (MUSAN-Dataset) |
| **Room Impulse Response** | Halleffekte simulieren (RIR-Dataset) |
| **Pitch Shifting** | Tonhöhe leicht variieren |
| **Volume Perturbation** | Lautstärke variieren |

---

## 6. Modellarchitekturen (Optionen)

### Option A: Conformer-CTC (Empfohlen für Eigenentwicklung)

```
Audio → Mel-Spec → Conv Subsampling → Conformer Blocks × N → Linear → CTC Loss
                                                                    ↓
                                                              Greedy / Beam Search → Text
```

**Conformer Block:**
```
Input → Feed Forward (½) → Multi-Head Self-Attention → Convolution Module → Feed Forward (½) → Output
```

Parameter:
- Encoder Layers: 12–18
- Attention Heads: 4–8
- Hidden Dim: 256–512
- Conv Kernel Size: 31
- Subsampling Factor: 4×

### Option B: Wav2Vec2 + Fine-Tuning (Schnellster Start)

```
Raw Audio → CNN Feature Encoder → Transformer Encoder → Linear → CTC Loss
            (pretrained)           (pretrained)         (fine-tuned)
```

- Pretrained Model: `facebook/wav2vec2-large-960h-lv60-self`
- Fine-Tuning auf eigenem Datensatz mit CTC Loss
- Wenig gelabelte Daten nötig (10–100h reichen)

### Option C: Whisper-Style (Encoder-Decoder)

```
Audio → Log-Mel Spec → Encoder (Transformer) → Decoder (Transformer) → Text Tokens
```

- Multitask: STT + Spracherkennung + Übersetzung
- Attention-basierter Decoder (autoregressive Generierung)
- Sehr gute Genauigkeit, aber langsamer als CTC

### Option D: Eigenes Modell von Grund auf (DeepSpeech2-Style)

```
Audio → Mel-Spec → Conv2D Layers → Bidirectional RNN (GRU/LSTM) → FC → CTC Loss
```

- Einfacher zu implementieren
- Guter Einstieg zum Lernen
- Geringere Accuracy als Conformer/Wav2Vec2

### Empfohlene Strategie

1. **Phase 1:** Starte mit **Wav2Vec2 Fine-Tuning** (schnelle Ergebnisse)
2. **Phase 2:** Baue **Conformer-CTC** von Grund auf (echte Eigenentwicklung)
3. **Phase 3:** Erweitere zu **Conformer-CTC/Attention Hybrid** (beste Qualität)

---

## 7. Audio-Preprocessing Pipeline

### 7.1 Vollständige Pipeline

```python
# Pseudocode der Preprocessing Pipeline

class AudioPreprocessor:
    def __init__(self):
        self.target_sr = 16000
        self.n_mels = 80
        self.win_length = 400      # 25ms bei 16kHz
        self.hop_length = 160      # 10ms bei 16kHz
        self.n_fft = 512

    def process(self, audio_path):
        # 1. Audio laden
        waveform, sr = load_audio(audio_path)

        # 2. Mono konvertieren
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 3. Resampling auf 16kHz
        if sr != self.target_sr:
            waveform = resample(waveform, sr, self.target_sr)

        # 4. Normalisierung
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)

        # 5. Mel-Spektrogramm
        mel_spec = compute_mel_spectrogram(
            waveform,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length
        )

        # 6. Log-Kompression
        log_mel = torch.log(mel_spec + 1e-9)

        return log_mel
```

### 7.2 Tokenization

Text muss in Tokens umgewandelt werden:

| Methode | Beschreibung | Vocab-Größe |
|---------|-------------|-------------|
| **Character-Level** | Jedes Zeichen = 1 Token (a, b, c, ...) | ~30–50 |
| **BPE (Byte Pair Encoding)** | Subword-Tokenization | 500–5000 |
| **Unigram (SentencePiece)** | Probabilistisches Subword-Modell | 500–5000 |
| **Word-Level** | Ganzes Wort = 1 Token | 50.000+ |

**Empfehlung:**
- **Character-Level** für CTC-Modelle (einfach, funktioniert gut)
- **BPE/Unigram** für Attention-basierte Decoder (effizienter)

---

## 8. Training Pipeline

### 8.1 Training-Konfiguration

```yaml
# Beispiel Training Config
model:
  type: conformer_ctc
  encoder_layers: 12
  attention_heads: 4
  hidden_dim: 256
  conv_kernel_size: 31
  dropout: 0.1

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  warmup_steps: 10000
  scheduler: noam  # oder cosine
  optimizer: adam
  weight_decay: 0.0001
  gradient_clip: 5.0
  accumulation_steps: 4
  mixed_precision: true  # fp16 Training

data:
  train_manifest: data/train.json
  val_manifest: data/val.json
  max_duration: 20.0     # Max Audio-Länge in Sekunden
  min_duration: 0.5
  augmentation:
    spec_augment: true
    speed_perturb: [0.9, 1.0, 1.1]
    noise_inject: true

ctc:
  blank_id: 0
  reduction: mean
```

### 8.2 Training Loop (Pseudocode)

```python
# Vereinfachter Training Loop

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        # Forward pass
        audio_features = preprocess(batch.audio)
        logits = model(audio_features)

        # CTC Loss berechnen
        log_probs = F.log_softmax(logits, dim=-1)
        loss = ctc_loss(
            log_probs=log_probs,
            targets=batch.tokens,
            input_lengths=batch.audio_lengths,
            target_lengths=batch.token_lengths
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

    # Validation
    model.eval()
    wer = evaluate(model, val_dataloader)
    log(f"Epoch {epoch}: WER = {wer:.2%}")
```

### 8.3 Wichtige Training-Tipps

1. **Curriculum Learning**: Starte mit kurzen Audio-Clips, erhöhe schrittweise
2. **SpecAugment**: Essentiell für Generalisierung (maskiere Frequenz- & Zeitbänder)
3. **Learning Rate Warmup**: Langsam starten, dann ansteigen
4. **Mixed Precision (FP16)**: Halbiert GPU-Speicherbedarf
5. **Gradient Accumulation**: Simuliert größere Batches
6. **Sortierung nach Länge**: Minimiert Padding innerhalb eines Batches

---

## 9. Decoding & Postprocessing

### 9.1 Decoding-Strategien

| Methode | Beschreibung | Geschwindigkeit | Qualität |
|---------|-------------|----------------|----------|
| **Greedy Decoding** | Nimmt pro Frame das wahrscheinlichste Zeichen | ⚡⚡⚡ | ⭐⭐ |
| **Beam Search** | Verfolgt die Top-K Hypothesen parallel | ⚡⚡ | ⭐⭐⭐ |
| **Beam Search + LM** | Beam Search mit n-gram Language Model Rescoring | ⚡ | ⭐⭐⭐⭐ |
| **Beam Search + LM + Hotwords** | Zusätzliche Gewichtung für wichtige Wörter | ⚡ | ⭐⭐⭐⭐⭐ |

### 9.2 CTC Decoding

```python
# Greedy CTC Decoding
def greedy_decode(logits, blank_id=0):
    # Argmax über Vocabulary
    predicted_ids = logits.argmax(dim=-1)

    # Duplikate zusammenführen
    decoded = []
    prev = blank_id
    for token_id in predicted_ids:
        if token_id != blank_id and token_id != prev:
            decoded.append(token_id)
        prev = token_id

    return tokens_to_text(decoded)
```

### 9.3 Beam Search mit Language Model

```python
# Beam Search mit KenLM
from pyctcdecode import build_ctcdecoder

decoder = build_ctcdecoder(
    labels=vocab_list,           # ["", "a", "b", "c", ...]
    kenlm_model_path="lm.arpa", # n-gram LM
    alpha=0.5,                   # LM Gewichtung
    beta=1.0,                    # Wort-Bonus
)

# Decoding
text = decoder.decode(logits_numpy, beam_width=100)
```

### 9.4 Postprocessing

| Schritt | Beschreibung | Tool/Methode |
|---------|-------------|-------------|
| **Groß/Kleinschreibung** | Erster Buchstabe groß, Eigennamen | Regex + kleines Modell |
| **Interpunktion** | Punkte, Kommas, Fragezeichen einfügen | Punctuation Model (BERT-basiert) |
| **Zahlen** | "dreiundzwanzig" → "23" | Inverse Text Normalization (ITN) |
| **Entitäten** | Datumsangaben, Uhrzeiten formatieren | Regel-basiert + NER |

---

## 10. Evaluation & Metriken

### 10.1 Hauptmetrik: Word Error Rate (WER)

$$WER = \frac{S + D + I}{N}$$

Wobei:
- $S$ = Substitutionen (falsche Wörter)
- $D$ = Deletions (fehlende Wörter)
- $I$ = Insertions (zusätzliche Wörter)
- $N$ = Gesamtanzahl Referenz-Wörter

### 10.2 Weitere Metriken

| Metrik | Beschreibung | Formel |
|--------|-------------|--------|
| **CER** | Character Error Rate | Wie WER, aber auf Zeichenebene |
| **WRR** | Word Recognition Rate | $WRR = 1 - WER$ |
| **RTF** | Real-Time Factor | Verarbeitungszeit / Audio-Dauer |
| **Latenz** | Zeit bis erstes Ergebnis | Wichtig für Streaming |

### 10.3 Benchmark-Ziele

| Dataset | Gute WER | State-of-the-Art WER |
|---------|---------|---------------------|
| LibriSpeech test-clean | < 5% | ~1.5% |
| LibriSpeech test-other | < 10% | ~3% |
| Common Voice DE test | < 15% | ~5% |

### 10.4 Evaluierungs-Script

```python
import jiwer

def evaluate(model, test_dataloader, decoder):
    all_references = []
    all_hypotheses = []

    for batch in test_dataloader:
        logits = model(batch.audio_features)
        hypotheses = decoder.batch_decode(logits)

        all_references.extend(batch.texts)
        all_hypotheses.extend(hypotheses)

    wer = jiwer.wer(all_references, all_hypotheses)
    cer = jiwer.cer(all_references, all_hypotheses)

    return {"wer": wer, "cer": cer}
```

---

## 11. Deployment & API

### 11.1 Model Export & Optimierung

| Methode | Beschreibung | Speedup |
|---------|-------------|---------|
| **ONNX Export** | Framework-unabhängiges Format | 2-3× |
| **TorchScript** | PyTorch JIT Compilation | 1.5-2× |
| **Quantisierung (INT8)** | Gewichte auf 8-Bit reduzieren | 2-4× |
| **Pruning** | Unwichtige Gewichte entfernen | 1.5-3× |
| **CTranslate2** | Optimierte Inference Library | 3-5× |

### 11.2 REST API (FastAPI)

```python
from fastapi import FastAPI, UploadFile
import uvicorn

app = FastAPI(title="STT Engine API")

@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    # Audio laden
    audio_bytes = await audio.read()
    waveform = load_audio_from_bytes(audio_bytes)

    # Preprocessing
    features = preprocessor.process(waveform)

    # Inference
    logits = model(features)
    text = decoder.decode(logits)

    return {
        "text": text,
        "confidence": confidence_score,
        "duration": audio_duration
    }

# WebSocket für Streaming
@app.websocket("/stream")
async def stream_transcribe(websocket):
    async for audio_chunk in websocket:
        partial_text = streaming_decoder.process_chunk(audio_chunk)
        await websocket.send_json({"partial": partial_text})
```

### 11.3 Streaming-Architektur (für Echtzeit)

```
Mikrofon → Audio Chunks (320ms) → Feature Extraction → Encoder → Decoder → Partial Text
                                                                              ↓
                                                                    Finaler Text (bei Stille)
```

---

## 12. Projektstruktur

```
STT/
├── STT_ENGINE_PLAN.md          # Dieser Plan
├── config/
│   ├── model_config.yaml       # Modell-Konfiguration
│   ├── train_config.yaml       # Training-Konfiguration
│   └── inference_config.yaml   # Inference-Konfiguration
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── audio_loader.py     # Audio I/O (WAV, FLAC, MP3)
│   │   ├── feature_extractor.py # Mel-Spektrogramm, MFCC
│   │   ├── augmentation.py     # SpecAugment, Noise, Speed
│   │   └── tokenizer.py        # Character/BPE Tokenizer
│   ├── model/
│   │   ├── __init__.py
│   │   ├── encoder.py          # Conformer/Transformer Encoder
│   │   ├── decoder.py          # CTC/Attention Decoder
│   │   ├── conformer.py        # Conformer Block Implementation
│   │   ├── subsampling.py      # Conv Subsampling Layer
│   │   └── model.py            # Gesamt-Modell
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training Loop
│   │   ├── scheduler.py        # Learning Rate Scheduler
│   │   ├── loss.py             # CTC Loss, Joint CTC-Att Loss
│   │   └── callbacks.py        # Checkpointing, Logging
│   ├── decoding/
│   │   ├── __init__.py
│   │   ├── greedy.py           # Greedy CTC Decoding
│   │   ├── beam_search.py      # Beam Search (mit/ohne LM)
│   │   └── streaming.py        # Streaming Decoder
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   ├── punctuation.py      # Interpunktion einfügen
│   │   ├── capitalization.py   # Groß/Kleinschreibung
│   │   └── normalization.py    # Inverse Text Normalization
│   └── api/
│       ├── __init__.py
│       ├── server.py           # FastAPI Server
│       └── websocket.py        # WebSocket Streaming
├── data/
│   ├── raw/                    # Originale Audio-Dateien
│   ├── processed/              # Vorverarbeitete Features
│   ├── manifests/              # JSON Manifest-Dateien
│   └── lm/                     # Language Model Dateien
├── models/
│   ├── checkpoints/            # Training Checkpoints
│   ├── exported/               # ONNX/TorchScript Modelle
│   └── pretrained/             # Heruntergeladene Pretrained Modelle
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_decoding.py
│   └── test_api.py
├── scripts/
│   ├── prepare_data.py         # Daten herunterladen & vorbereiten
│   ├── train.py                # Training starten
│   ├── evaluate.py             # Evaluation auf Testset
│   ├── export_model.py         # Modell exportieren
│   └── transcribe.py           # Einzelne Datei transkribieren
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_experiments.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

---

## 13. Implementierungs-Phasen

### Phase 1: Grundlagen & Quick Win (Woche 1-2)

- [ ] Projektstruktur anlegen
- [ ] Audio-Loader implementieren (WAV/FLAC laden, resampling)
- [ ] Feature Extraction (Mel-Spektrogramm) implementieren
- [ ] Character-Level Tokenizer bauen
- [ ] LibriSpeech Dataset herunterladen und Manifest erstellen
- [ ] **Quick Win:** Wav2Vec2 Fine-Tuning mit HuggingFace (sofort funktionsfähige STT)

### Phase 2: Eigenes Modell (Woche 3-5)

- [ ] Conv Subsampling Layer implementieren
- [ ] Conformer Block implementieren (Self-Attention + Convolution + FFN)
- [ ] Gesamten Conformer Encoder zusammenbauen
- [ ] CTC Loss Integration
- [ ] Greedy CTC Decoder
- [ ] Training Loop mit Gradient Clipping, LR Scheduler
- [ ] SpecAugment Datenaugmentierung

### Phase 3: Verbesserung (Woche 6-8)

- [ ] Beam Search Decoder implementieren
- [ ] KenLM Language Model trainieren und integrieren
- [ ] Attention Decoder (optional) für CTC/Attention Hybrid
- [ ] Speed Perturbation & Noise Augmentation
- [ ] Mixed Precision Training (FP16)
- [ ] WER Evaluation Pipeline

### Phase 4: Polish & Deployment (Woche 9-10)

- [ ] Postprocessing (Interpunktion, Kapitalisierung)
- [ ] ONNX Export für schnelle Inference
- [ ] FastAPI REST-API bauen
- [ ] WebSocket Streaming-Endpoint
- [ ] Docker Container für Deployment
- [ ] Dokumentation & Tests

### Phase 5: Erweitert (Optional)

- [ ] Deutsches Modell trainieren (Common Voice DE)
- [ ] Speaker Diarization ("Wer spricht wann?")
- [ ] Multilingual Support
- [ ] On-Device Inference (Quantisierung, ONNX Mobile)
- [ ] Keyword Spotting / Wake Word Detection

---

## 14. Hardware-Anforderungen

### Minimum (für Fine-Tuning & kleine Modelle)

| Komponente | Empfehlung |
|-----------|-----------|
| **GPU** | NVIDIA RTX 3060 (12 GB VRAM) |
| **RAM** | 32 GB |
| **Storage** | 500 GB SSD (für Datasets) |
| **CPU** | 8+ Kerne (für Daten-Loading) |

### Empfohlen (für Training von Grund auf)

| Komponente | Empfehlung |
|-----------|-----------|
| **GPU** | NVIDIA RTX 4090 (24 GB) oder A100 (40/80 GB) |
| **RAM** | 64 GB+ |
| **Storage** | 2 TB NVMe SSD |
| **CPU** | 16+ Kerne |

### Cloud-Alternativen

| Anbieter | GPU | Kosten ca. |
|----------|-----|-----------|
| **Google Colab Pro+** | A100/V100 | ~50€/Monat |
| **Lambda Labs** | A100×1 | ~1.10$/h |
| **RunPod** | A100/H100 | ab 1.10$/h |
| **Vast.ai** | Diverse | ab 0.30$/h |

---

## 15. Bekannte Herausforderungen

| Herausforderung | Lösung |
|----------------|--------|
| **Lange Audio-Dateien** | Chunking in 10-30s Segmente, Overlapping |
| **Verschiedene Akzente/Dialekte** | Diverse Trainingsdaten, Data Augmentation |
| **Hintergrundgeräusche** | Noise Augmentation, Noise-Robust Features |
| **Fachbegriffe/Eigennamen** | Hotword-Boosting, domänenspezifisches LM |
| **Interpunktion fehlt** | Separates Punctuation Model nachschalten |
| **GPU-Speicher bei langen Sequenzen** | Gradient Checkpointing, Chunked Attention |
| **Out-of-Vocabulary Wörter** | BPE/Subword Tokenization |
| **Streaming/Echtzeit** | RNN-T oder Chunk-basiertes Conformer |

---

## 16. Referenzen & Ressourcen

### Schlüssel-Paper

| Paper | Jahr | Thema |
|-------|------|-------|
| **Attention Is All You Need** (Vaswani et al.) | 2017 | Transformer-Architektur |
| **wav2vec 2.0** (Baevski et al.) | 2020 | Self-Supervised Pretraining für Speech |
| **Conformer** (Gulati et al.) | 2020 | Convolution-augmented Transformer |
| **CTC** (Graves et al.) | 2006 | Connectionist Temporal Classification |
| **Listen, Attend and Spell** (Chan et al.) | 2016 | Attention-basierte STT |
| **Deep Speech 2** (Amodei et al.) | 2016 | End-to-End STT in EN + Mandarin |
| **Whisper** (Radford et al.) | 2022 | Robuste multilingual STT |
| **SpecAugment** (Park et al.) | 2019 | Datenaugmentierung für STT |

### Open-Source STT Frameworks (zum Lernen)

| Projekt | Sprache | Beschreibung |
|---------|---------|-------------|
| **ESPnet** | Python | Umfassendstes E2E STT Framework |
| **NeMo** (NVIDIA) | Python | Production-grade STT/TTS |
| **Whisper** (OpenAI) | Python | Multilingual STT Modell |
| **Vosk** | C++/Python | Offline-fähiges STT |
| **Coqui STT** | Python | Fork von Mozilla DeepSpeech |
| **WeNet** | C++/Python | Production-ready Conformer |
| **k2/icefall** | Python | Nächste Generation von Kaldi |

### Tutorials & Guides

- HuggingFace: Fine-Tuning Wav2Vec2 für ASR
- PyTorch Audio: ASR Inference mit CTC Decoder
- ESPnet: Conformer Training Recipes
- NVIDIA NeMo: ASR Tutorial

---

## Zusammenfassung: Empfohlener Startpfad

```
1. Wav2Vec2 Fine-Tuning (Tag 1-3)
   → Sofort funktionierende STT mit pretrained Modell
   → Lernen der grundlegenden Pipeline

2. Eigene Preprocessing-Pipeline (Tag 4-7)
   → Audio Loader, Mel-Spektrogramm, Tokenizer
   → Verständnis der Datenaufbereitung

3. Conformer-CTC Modell bauen (Tag 8-21)
   → Eigene Encoder-Architektur
   → CTC Training auf LibriSpeech

4. Decoder & LM Integration (Tag 22-28)
   → Beam Search + KenLM
   → Deutliche WER-Verbesserung

5. API & Deployment (Tag 29-35)
   → FastAPI REST Endpoint
   → ONNX Export für schnelle Inference
```

> **Wichtig:** Der schnellste Weg zu einem funktionierenden Prototyp ist Wav2Vec2 Fine-Tuning (Phase 1). Der Aufbau eines eigenen Conformer-Modells von Grund auf (Phase 2+) braucht mehr Zeit, gibt aber maximales Verständnis und Kontrolle.
