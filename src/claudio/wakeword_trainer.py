"""Wake word model training using synthetic TTS and user samples.

Uses claudio's TTS backends (kokoro, soprano, say) for synthetic positive
sample generation, combined with user recordings and data augmentation.

Training pipeline:
1. Generate synthetic samples with multiple TTS voices
2. Record user voice samples (optional, improves accuracy)
3. Augment data with noise, pitch/tempo variations, reverb
4. Train classifier on Google speech embeddings
5. Export to ONNX for use with openWakeWord

Example:
```python
from claudio.wakeword_trainer import WakeWordTrainer

trainer = WakeWordTrainer('hey_claude')
trainer.generate_synthetic(count=5000)
trainer.record_samples(count=20)  # Interactive recording
trainer.augment()
trainer.train()
trainer.export('models/hey_claude.onnx')
```
"""

from __future__ import annotations

import io
import json
import random
import subprocess
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


@dataclass
class TrainerConfig:
  """Configuration for wake word training."""
  wake_word: str
  output_dir: Path = field(default_factory=lambda: Path('wakeword_training'))

  # TTS settings - uses claudio's TTS backends
  tts_backend: str = 'auto'  # 'auto', 'kokoro', 'soprano', 'say'
  tts_voices: list[str] = field(default_factory=list)  # Voice names to cycle through
  synthetic_count: int = 5000

  # Recording settings
  sample_rate: int = 16000
  record_duration: float = 2.0

  # Augmentation settings
  add_noise: bool = True
  noise_levels: list[float] = field(default_factory=lambda: [0.001, 0.005, 0.01])
  pitch_shift: bool = True
  pitch_range: tuple[float, float] = (-2.0, 2.0)  # semitones
  tempo_shift: bool = True
  tempo_range: tuple[float, float] = (0.9, 1.1)
  add_reverb: bool = True

  # Training settings
  epochs: int = 100
  batch_size: int = 64
  learning_rate: float = 0.001
  validation_split: float = 0.1

  # Model settings
  model_type: str = 'dnn'  # 'dnn' or 'gru'
  hidden_size: int = 64


class SyntheticGenerator:
  """Generate synthetic wake word samples using claudio's TTS."""

  def __init__(self, config: TrainerConfig):
    self.config = config
    self.output_dir = config.output_dir / 'synthetic'
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self._tts = None
    self._available_voices: list[dict] | None = None

  def _get_tts(self):
    """Get or create TTS instance."""
    if self._tts is None:
      try:
        from claudio import tts

        tts_config = tts.Config(backend=self.config.tts_backend)
        self._tts = tts.create(tts_config)

        # Get available voices
        try:
          self._available_voices = self._tts.voices()
          if self._available_voices:
            voice_names = [v.get('name', v.get('id', '?')) for v in self._available_voices[:5]]
            console.print(f'[dim]Available voices: {", ".join(voice_names)}...[/dim]')
        except Exception:
          self._available_voices = []

      except ImportError as e:
        console.print(f'[yellow]TTS not available: {e}[/yellow]')
        return None

    return self._tts

  def _generate_with_claudio_tts(
    self,
    text: str,
    output_path: Path,
    voice: str | None = None,
    rate: float = 1.0,
  ) -> bool:
    """Generate audio using claudio's TTS."""
    tts = self._get_tts()
    if tts is None:
      return False

    try:
      # Generate audio bytes (WAV format)
      audio_bytes = tts.generate(text, voice=voice, rate=rate)

      if audio_bytes and len(audio_bytes) > 0:
        # Write directly to file
        with open(output_path, 'wb') as f:
          f.write(audio_bytes)

        # Resample to 16kHz if needed for training
        self._resample_if_needed(output_path)
        return True

    except Exception as e:
      console.print(f'[dim]TTS error: {e}[/dim]')

    return False

  def _resample_if_needed(self, path: Path):
    """Resample audio to 16kHz if needed for training."""
    try:
      with wave.open(str(path), 'rb') as wf:
        sample_rate = wf.getframerate()
        if sample_rate == self.config.sample_rate:
          return  # Already correct rate

        # Read audio
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        channels = wf.getnchannels()

      # Convert to mono if stereo
      if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

      # Resample using linear interpolation
      if sample_rate != self.config.sample_rate:
        duration = len(audio) / sample_rate
        new_length = int(duration * self.config.sample_rate)
        indices = np.linspace(0, len(audio) - 1, new_length)
        audio = np.interp(indices, np.arange(len(audio)), audio)

      # Write back
      audio_int = audio.astype(np.int16)
      with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.config.sample_rate)
        wf.writeframes(audio_int.tobytes())

    except Exception as e:
      console.print(f'[dim]Resample error: {e}[/dim]')

  def _get_voice_for_sample(self, index: int) -> str | None:
    """Get voice to use for a sample (cycles through configured voices)."""
    # Use configured voices if available
    if self.config.tts_voices:
      return self.config.tts_voices[index % len(self.config.tts_voices)]

    # Otherwise cycle through available voices
    if self._available_voices:
      voice_info = self._available_voices[index % len(self._available_voices)]
      return voice_info.get('name') or voice_info.get('id')

    return None

  def _get_rate_variation(self) -> float:
    """Get random rate variation for diversity."""
    return random.uniform(0.85, 1.15)

  def generate(
    self,
    count: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
  ) -> int:
    """Generate synthetic samples.

    Returns number of samples generated.
    """
    count = count or self.config.synthetic_count
    generated = 0
    failed_streak = 0

    # Variations of the wake word
    text_variations = self._get_text_variations()

    console.print(f'[cyan]Generating {count} synthetic samples...[/cyan]')

    with Progress(
      SpinnerColumn(),
      TextColumn('[progress.description]{task.description}'),
      BarColumn(),
      TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
      console=console,
    ) as progress:
      task = progress.add_task('Generating...', total=count)

      while generated < count:
        text = random.choice(text_variations)
        output_path = self.output_dir / f'synthetic_{generated:06d}.wav'

        voice = self._get_voice_for_sample(generated)
        rate = self._get_rate_variation()

        success = self._generate_with_claudio_tts(text, output_path, voice, rate)

        if success:
          generated += 1
          failed_streak = 0
          progress.update(task, completed=generated)
          if progress_callback:
            progress_callback(generated, count)
        else:
          failed_streak += 1
          if failed_streak >= 5:
            console.print('[red]TTS generation failing repeatedly[/red]')
            console.print('[dim]Check that TTS is properly configured[/dim]')
            break

    console.print(f'[green]Generated {generated} synthetic samples[/green]')
    return generated

  def _get_text_variations(self) -> list[str]:
    """Get variations of the wake word for diverse training."""
    wake_word = self.config.wake_word

    # Basic variations
    variations = [
      wake_word,
      wake_word.lower(),
      wake_word.upper(),
      wake_word.capitalize(),
    ]

    # Add punctuation variations
    for base in list(variations):
      variations.extend([
        f'{base}.',
        f'{base}!',
        f'{base}?',
        f'{base},',
      ])

    # Add context variations
    contexts = [
      '{word}',
      '{word} please',
      'Okay {word}',
      'Um {word}',
      'Hey {word}',
      '{word} help me',
      '{word} what time is it',
      '{word} turn on the lights',
    ]

    for ctx in contexts:
      variations.append(ctx.format(word=wake_word))

    return list(set(variations))


class SampleRecorder:
  """Record user voice samples for training."""

  def __init__(self, config: TrainerConfig):
    self.config = config
    self.output_dir = config.output_dir / 'user_samples'
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self._recording = False

  def record_one(self, index: int, countdown: int = 3) -> Path | None:
    """Record a single sample with countdown.

    Returns path to recorded file or None if failed.
    """
    try:
      import pyaudio
    except ImportError:
      console.print('[yellow]pyaudio not installed[/yellow]')
      return None

    output_path = self.output_dir / f'user_{index:04d}.wav'

    # Countdown
    for i in range(countdown, 0, -1):
      console.print(f'[dim]Recording in {i}...[/dim]', end='\r')
      time.sleep(1)

    console.print(f'[red]Recording... Say: "{self.config.wake_word}"[/red]')

    # Record
    audio = pyaudio.PyAudio()
    frames = []

    try:
      stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=self.config.sample_rate,
        input=True,
        frames_per_buffer=1024,
      )

      num_frames = int(
        self.config.sample_rate * self.config.record_duration / 1024
      )

      for _ in range(num_frames):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

      stream.stop_stream()
      stream.close()

    finally:
      audio.terminate()

    # Save
    with wave.open(str(output_path), 'wb') as wf:
      wf.setnchannels(1)
      wf.setsampwidth(2)
      wf.setframerate(self.config.sample_rate)
      wf.writeframes(b''.join(frames))

    console.print(f'[green]Saved: {output_path.name}[/green]')
    return output_path

  def record_batch(
    self,
    count: int = 20,
    auto_continue: bool = False,
  ) -> list[Path]:
    """Record multiple samples interactively.

    Returns list of recorded file paths.
    """
    recorded = []

    console.print()
    console.print(f'[bold]Recording {count} samples of "{self.config.wake_word}"[/bold]')
    console.print('[dim]Press Enter to record, or q to quit[/dim]')
    console.print()

    for i in range(count):
      if not auto_continue:
        response = input(f'Sample {i+1}/{count} - Press Enter to record (q to quit): ')
        if response.lower() == 'q':
          break

      path = self.record_one(i)
      if path:
        recorded.append(path)

      if auto_continue and i < count - 1:
        time.sleep(0.5)

    console.print()
    console.print(f'[green]Recorded {len(recorded)} samples[/green]')
    return recorded


class DataAugmenter:
  """Augment training data with variations."""

  def __init__(self, config: TrainerConfig):
    self.config = config
    self.input_dirs = [
      config.output_dir / 'synthetic',
      config.output_dir / 'user_samples',
    ]
    self.output_dir = config.output_dir / 'augmented'
    self.output_dir.mkdir(parents=True, exist_ok=True)

  def _load_audio(self, path: Path) -> tuple[np.ndarray, int] | None:
    """Load audio file as numpy array."""
    try:
      with wave.open(str(path), 'rb') as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        return audio, sample_rate
    except Exception:
      return None

  def _save_audio(self, audio: np.ndarray, path: Path, sample_rate: int):
    """Save numpy array as audio file."""
    audio_int = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
      wf.setnchannels(1)
      wf.setsampwidth(2)
      wf.setframerate(sample_rate)
      wf.writeframes(audio_int.tobytes())

  def _add_noise(self, audio: np.ndarray, level: float) -> np.ndarray:
    """Add gaussian noise to audio."""
    noise = np.random.normal(0, level, len(audio))
    return audio + noise

  def _shift_pitch(self, audio: np.ndarray, semitones: float, sample_rate: int) -> np.ndarray:
    """Shift pitch by semitones (requires librosa)."""
    try:
      import librosa
      return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)
    except ImportError:
      return audio

  def _shift_tempo(self, audio: np.ndarray, rate: float) -> np.ndarray:
    """Shift tempo by rate factor (requires librosa)."""
    try:
      import librosa
      return librosa.effects.time_stretch(audio, rate=rate)
    except ImportError:
      return audio

  def _add_reverb(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Add simple reverb effect."""
    # Simple comb filter reverb
    delay_samples = int(0.03 * sample_rate)  # 30ms delay
    decay = 0.3

    output = np.zeros(len(audio) + delay_samples)
    output[:len(audio)] = audio

    for i in range(delay_samples, len(output)):
      output[i] += decay * output[i - delay_samples]

    return output[:len(audio)]

  def augment(self, multiplier: int = 5) -> int:
    """Augment all samples.

    Args:
      multiplier: Number of augmented versions per original sample

    Returns number of augmented samples created.
    """
    # Collect input files
    input_files = []
    for input_dir in self.input_dirs:
      if input_dir.exists():
        input_files.extend(input_dir.glob('*.wav'))

    if not input_files:
      console.print('[yellow]No input files found for augmentation[/yellow]')
      return 0

    console.print(f'[cyan]Augmenting {len(input_files)} samples (x{multiplier})...[/cyan]')

    created = 0

    with Progress(
      SpinnerColumn(),
      TextColumn('[progress.description]{task.description}'),
      BarColumn(),
      console=console,
    ) as progress:
      task = progress.add_task('Augmenting...', total=len(input_files) * multiplier)

      for input_path in input_files:
        result = self._load_audio(input_path)
        if result is None:
          continue

        audio, sample_rate = result
        base_name = input_path.stem

        for i in range(multiplier):
          aug_audio = audio.copy()
          suffix_parts = []

          # Random augmentations
          if self.config.add_noise and random.random() < 0.7:
            level = random.choice(self.config.noise_levels)
            aug_audio = self._add_noise(aug_audio, level)
            suffix_parts.append(f'n{level:.3f}')

          if self.config.pitch_shift and random.random() < 0.5:
            semitones = random.uniform(*self.config.pitch_range)
            aug_audio = self._shift_pitch(aug_audio, semitones, sample_rate)
            suffix_parts.append(f'p{semitones:+.1f}')

          if self.config.tempo_shift and random.random() < 0.5:
            rate = random.uniform(*self.config.tempo_range)
            aug_audio = self._shift_tempo(aug_audio, rate)
            suffix_parts.append(f't{rate:.2f}')

          if self.config.add_reverb and random.random() < 0.3:
            aug_audio = self._add_reverb(aug_audio, sample_rate)
            suffix_parts.append('r')

          # Save augmented version
          suffix = '_'.join(suffix_parts) if suffix_parts else f'v{i}'
          output_path = self.output_dir / f'{base_name}_{suffix}.wav'
          self._save_audio(aug_audio, output_path, sample_rate)
          created += 1
          progress.update(task, advance=1)

    console.print(f'[green]Created {created} augmented samples[/green]')
    return created


class ModelTrainer:
  """Train wake word detection model."""

  def __init__(self, config: TrainerConfig):
    self.config = config
    self.data_dir = config.output_dir / 'augmented'
    self.model_dir = config.output_dir / 'model'
    self.model_dir.mkdir(parents=True, exist_ok=True)

  def prepare_features(self) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract features from audio files using Google embeddings."""
    try:
      import openwakeword
      from openwakeword.model import Model
    except ImportError:
      console.print('[yellow]openwakeword not installed[/yellow]')
      console.print('[dim]Install with: uv sync --extra triggers[/dim]')
      return None

    # Get embedding model
    try:
      embedding_model = openwakeword.utils.get_embedding_model()
    except Exception as e:
      console.print(f'[red]Failed to load embedding model: {e}[/red]')
      return None

    # Collect audio files
    audio_files = list(self.data_dir.glob('*.wav'))
    if not audio_files:
      # Also check synthetic and user_samples if augmented is empty
      for subdir in ['synthetic', 'user_samples']:
        subpath = self.config.output_dir / subdir
        if subpath.exists():
          audio_files.extend(subpath.glob('*.wav'))

    if not audio_files:
      console.print('[yellow]No audio files found for training[/yellow]')
      return None

    console.print(f'[cyan]Extracting features from {len(audio_files)} files...[/cyan]')

    features = []
    labels = []

    with Progress(
      SpinnerColumn(),
      TextColumn('[progress.description]{task.description}'),
      BarColumn(),
      console=console,
    ) as progress:
      task = progress.add_task('Extracting...', total=len(audio_files))

      for audio_path in audio_files:
        try:
          # Load audio
          with wave.open(str(audio_path), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)

          # Get embeddings (process in chunks)
          chunk_size = 1280  # 80ms at 16kHz
          for i in range(0, len(audio) - chunk_size, chunk_size):
            chunk = audio[i:i + chunk_size]
            embedding = embedding_model.predict(chunk)
            features.append(embedding)
            labels.append(1)  # Positive sample

        except Exception as e:
          console.print(f'[dim]Error processing {audio_path.name}: {e}[/dim]')

        progress.update(task, advance=1)

    if not features:
      console.print('[yellow]No features extracted[/yellow]')
      return None

    X = np.array(features)
    y = np.array(labels)

    console.print(f'[green]Extracted {len(features)} feature vectors[/green]')
    return X, y

  def train(self, X: np.ndarray, y: np.ndarray) -> bool:
    """Train the classifier model."""
    try:
      import torch
      import torch.nn as nn
      from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
      console.print('[yellow]PyTorch not installed[/yellow]')
      return False

    console.print('[cyan]Training classifier...[/cyan]')

    # Split data
    split_idx = int(len(X) * (1 - self.config.validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Create datasets
    train_dataset = TensorDataset(
      torch.FloatTensor(X_train),
      torch.FloatTensor(y_train),
    )
    val_dataset = TensorDataset(
      torch.FloatTensor(X_val),
      torch.FloatTensor(y_val),
    )

    train_loader = DataLoader(
      train_dataset,
      batch_size=self.config.batch_size,
      shuffle=True,
    )
    val_loader = DataLoader(
      val_dataset,
      batch_size=self.config.batch_size,
    )

    # Build model
    input_size = X.shape[1]

    if self.config.model_type == 'gru':
      model = nn.Sequential(
        nn.GRU(input_size, self.config.hidden_size, batch_first=True),
        nn.Linear(self.config.hidden_size, 1),
        nn.Sigmoid(),
      )
    else:  # DNN
      model = nn.Sequential(
        nn.Linear(input_size, self.config.hidden_size),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(self.config.hidden_size // 2, 1),
        nn.Sigmoid(),
      )

    # Training
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

    best_val_loss = float('inf')

    with Progress(
      SpinnerColumn(),
      TextColumn('[progress.description]{task.description}'),
      BarColumn(),
      console=console,
    ) as progress:
      task = progress.add_task('Training...', total=self.config.epochs)

      for epoch in range(self.config.epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
          optimizer.zero_grad()
          outputs = model(batch_X).squeeze()
          loss = criterion(outputs, batch_y)
          loss.backward()
          optimizer.step()
          train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
          for batch_X, batch_y in val_loader:
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

        val_loss /= len(val_loader)

        # Save best model
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          torch.save(model.state_dict(), self.model_dir / 'best_model.pt')

        progress.update(
          task,
          advance=1,
          description=f'Training... (val_loss: {val_loss:.4f})',
        )

    console.print(f'[green]Training complete. Best val_loss: {best_val_loss:.4f}[/green]')

    # Save final model
    torch.save(model.state_dict(), self.model_dir / 'final_model.pt')
    torch.save(model, self.model_dir / 'model.pt')

    return True

  def export_onnx(self, output_path: Path | None = None) -> Path | None:
    """Export model to ONNX format."""
    try:
      import torch
    except ImportError:
      console.print('[yellow]PyTorch not installed[/yellow]')
      return None

    model_path = self.model_dir / 'model.pt'
    if not model_path.exists():
      console.print('[yellow]No trained model found[/yellow]')
      return None

    output_path = output_path or (self.model_dir / f'{self.config.wake_word}.onnx')

    console.print(f'[cyan]Exporting to ONNX: {output_path}[/cyan]')

    try:
      model = torch.load(model_path)
      model.eval()

      # Create dummy input
      dummy_input = torch.randn(1, 96)  # Google embedding size

      torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
          'input': {0: 'batch_size'},
          'output': {0: 'batch_size'},
        },
        opset_version=11,
      )

      console.print(f'[green]Exported: {output_path}[/green]')
      return output_path

    except Exception as e:
      console.print(f'[red]Export failed: {e}[/red]')
      return None


class WakeWordTrainer:
  """High-level interface for wake word training."""

  def __init__(
    self,
    wake_word: str,
    output_dir: Path | str | None = None,
  ):
    self.config = TrainerConfig(
      wake_word=wake_word,
      output_dir=Path(output_dir) if output_dir else Path(f'wakeword_{wake_word}'),
    )
    self.config.output_dir.mkdir(parents=True, exist_ok=True)

    self.generator = SyntheticGenerator(self.config)
    self.recorder = SampleRecorder(self.config)
    self.augmenter = DataAugmenter(self.config)
    self.trainer = ModelTrainer(self.config)

  def generate_synthetic(self, count: int | None = None) -> int:
    """Generate synthetic TTS samples."""
    return self.generator.generate(count)

  def record_samples(self, count: int = 20, auto_continue: bool = False) -> list[Path]:
    """Record user voice samples."""
    return self.recorder.record_batch(count, auto_continue)

  def augment(self, multiplier: int = 5) -> int:
    """Augment all collected samples."""
    return self.augmenter.augment(multiplier)

  def train(self) -> bool:
    """Train the wake word model."""
    result = self.trainer.prepare_features()
    if result is None:
      return False

    X, y = result
    return self.trainer.train(X, y)

  def export(self, output_path: Path | str | None = None) -> Path | None:
    """Export model to ONNX format."""
    if output_path:
      output_path = Path(output_path)
    return self.trainer.export_onnx(output_path)

  def run_full_pipeline(
    self,
    synthetic_count: int = 5000,
    user_samples: int = 20,
    augment_multiplier: int = 5,
  ) -> Path | None:
    """Run the complete training pipeline.

    Returns path to exported model or None if failed.
    """
    console.print()
    console.print(f'[bold blue]Wake Word Training: "{self.config.wake_word}"[/bold blue]')
    console.print(f'[dim]Output: {self.config.output_dir}[/dim]')
    console.print()

    # Step 1: Generate synthetic samples
    console.print('[bold]Step 1/5: Generate synthetic samples[/bold]')
    synthetic = self.generate_synthetic(synthetic_count)
    if synthetic == 0:
      console.print('[yellow]Warning: No synthetic samples generated[/yellow]')
    console.print()

    # Step 2: Record user samples (optional)
    if user_samples > 0:
      console.print('[bold]Step 2/5: Record user samples[/bold]')
      response = input(f'Record {user_samples} samples? [y/N]: ')
      if response.lower() == 'y':
        self.record_samples(user_samples)
      console.print()

    # Step 3: Augment data
    console.print('[bold]Step 3/5: Augment data[/bold]')
    augmented = self.augment(augment_multiplier)
    if augmented == 0 and synthetic == 0:
      console.print('[red]No data available for training[/red]')
      return None
    console.print()

    # Step 4: Train model
    console.print('[bold]Step 4/5: Train model[/bold]')
    if not self.train():
      console.print('[red]Training failed[/red]')
      return None
    console.print()

    # Step 5: Export model
    console.print('[bold]Step 5/5: Export model[/bold]')
    model_path = self.export()

    if model_path:
      console.print()
      console.print('[bold green]Training complete![/bold green]')
      console.print(f'Model saved to: {model_path}')
      console.print()
      console.print('[dim]To use the model, add to OPENWAKEWORD_MODELS in triggers.py[/dim]')

    return model_path

  def save_config(self, path: Path | None = None):
    """Save training configuration."""
    path = path or (self.config.output_dir / 'config.json')

    config_dict = {
      'wake_word': self.config.wake_word,
      'tts_voices': self.config.tts_voices,
      'synthetic_count': self.config.synthetic_count,
      'sample_rate': self.config.sample_rate,
      'epochs': self.config.epochs,
      'batch_size': self.config.batch_size,
      'model_type': self.config.model_type,
    }

    with open(path, 'w') as f:
      json.dump(config_dict, f, indent=2)

    console.print(f'[dim]Config saved: {path}[/dim]')

  @classmethod
  def load_config(cls, path: Path | str) -> 'WakeWordTrainer':
    """Load trainer from config file."""
    path = Path(path)

    with open(path) as f:
      config_dict = json.load(f)

    trainer = cls(config_dict['wake_word'], path.parent)

    for key, value in config_dict.items():
      if hasattr(trainer.config, key):
        setattr(trainer.config, key, value)

    return trainer
