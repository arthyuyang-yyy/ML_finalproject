"""Stage hparams.yaml for pyannote/segmentation@Interspeech2021.

The model was originally trained with pyannote.audio 0.0.1 (torch 1.10) and
its HF repo only ships the Lightning checkpoint. pyannote.audio 4.x needs a
hparams.yaml to instantiate the model class before loading the state_dict, so
we reconstruct it from the embedded hyper_parameters in the checkpoint and
write it into the standard HF cache layout.
"""
import hashlib
from pathlib import Path

CONTENT = """# Reconstructed from pytorch_model.bin (Lightning 1.5.4 checkpoint)
# embedded hyper_parameters and pyannote.audio metadata.
# Required by pyannote.audio 4.x Model.from_pretrained() to instantiate
# the architecture before loading the state_dict.
architecture: pyannote.audio.models.segmentation.PyanNet
sample_rate: 16000
num_channels: 1
sincnet:
  stride: 10
  sample_rate: 16000
lstm:
  hidden_size: 128
  num_layers: 4
  bidirectional: true
  monolithic: true
  dropout: 0.5
  batch_first: true
linear:
  hidden_size: 128
  num_layers: 2
task: overlapped_speech_detection
"""

SNAP = Path("/home/soulcode/.cache/huggingface/hub/models--pyannote--segmentation/snapshots/Interspeech2021")
BLOBS = Path("/home/soulcode/.cache/huggingface/hub/models--pyannote--segmentation/blobs")
dest = SNAP / "hparams.yaml"
dest.write_text(CONTENT)
h = hashlib.sha256(dest.read_bytes()).hexdigest()
blob = BLOBS / h
blob.write_bytes(dest.read_bytes())
dest.unlink()
dest.symlink_to(f"../../blobs/{h}")
print(f"hparams.yaml staged: {dest} -> {blob} sha256={h[:12]}")