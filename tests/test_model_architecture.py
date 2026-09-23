"""Regression tests for checkpoint-bound LSTM/GRU architecture selection."""

import unittest

import torch

from src.config import INPUT_FEATURES, SEQUENCE_LENGTH
from src.model import (
    FallDetectionModel,
    checkpoint_rnn_type,
    count_parameters,
)


class ModelArchitectureTests(unittest.TestCase):
    def test_gru_is_lighter_and_preserves_output_shape(self):
        lstm = FallDetectionModel(input_size=INPUT_FEATURES, rnn_type="LSTM")
        gru = FallDetectionModel(input_size=INPUT_FEATURES, rnn_type="GRU")
        self.assertLess(count_parameters(gru), count_parameters(lstm))
        values = torch.zeros(2, SEQUENCE_LENGTH, INPUT_FEATURES)
        for model in (lstm, gru):
            logits, attention = model(values)
            self.assertEqual(tuple(logits.shape), (2, 2))
            self.assertEqual(tuple(attention.shape), (2, SEQUENCE_LENGTH))

    def test_checkpoint_rnn_type_infers_legacy_and_gru_weights(self):
        lstm = FallDetectionModel(input_size=INPUT_FEATURES, rnn_type="LSTM")
        gru = FallDetectionModel(input_size=INPUT_FEATURES, rnn_type="GRU")
        self.assertEqual(
            checkpoint_rnn_type({"model_state_dict": lstm.state_dict()}),
            "LSTM",
        )
        self.assertEqual(
            checkpoint_rnn_type({"model_state_dict": gru.state_dict()}),
            "GRU",
        )

    def test_checkpoint_rejects_metadata_weight_mismatch(self):
        gru = FallDetectionModel(input_size=INPUT_FEATURES, rnn_type="GRU")
        with self.assertRaisesRegex(ValueError, "weight menunjukkan"):
            checkpoint_rnn_type({
                "training_config": {"rnn_type": "LSTM"},
                "model_state_dict": gru.state_dict(),
            })


if __name__ == "__main__":
    unittest.main()
