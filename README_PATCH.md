# LogicLSTM-inspired student patch

New modules only; existing spectral model files are not overwritten.

- `lstm_model.py`: raw-input, one-layer LSTM student, hidden size 128 by default; absolute-input relevance divided by total sample magnitude.
- `cache_teacher_probs.py`: generates cached teacher soft labels from official LogicLSTM ONNX for the exact deterministic subset in a config.
- `train_lstm.py`: trains either supervised custom LSTM or knowledge-distilled LSTM.
- `export_lstm_onnx.py`: exports the self-trained LSTM ONNX without embedded input normalization, matching the baseline's raw-input deployment form.
- three 100k configs: supervised, KD alpha 0.3 and KD alpha 0.6.

Unzip over the project root, then reinstall editable package if necessary: `python -m pip install -e .`.
