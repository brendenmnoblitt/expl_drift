"""Tests for Transformer attribution extractors.

These tests are hermetic: they build a tiny BERT entirely from an
in-memory config with random weights and a local vocab file, so no
network access or HuggingFace Hub download is required.  They verify
output shapes, padding masking, and compatibility with the downstream
drift-detection pipeline.

The whole module is skipped when the optional ``transformer`` extra
(``transformers``) is not installed.
"""

import numpy as np
import pytest

pytest.importorskip(
    "transformers",
    reason="requires the 'transformer' extra: pip install expl_drift[transformer]",
)

from transformers import (  # noqa: E402
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
)

from expl_drift.explanations.attention import explain_attention  # noqa: E402
from expl_drift.explanations.ig_transformer import explain_ig_transformer  # noqa: E402
from expl_drift.explanations.transformer import (  # noqa: E402
    get_classification_token_index,
    mask_padding,
    tokenize_texts,
)

MAX_LENGTH = 32
TEXTS = [
    "The stock market rose sharply today.",
    "Scientists discovered a new species of frog in the Amazon.",
    "The team won the championship game in overtime.",
]

# Minimal BERT vocabulary containing the special tokens plus a handful
# of common English words covering the fixture texts.  Anything not in
# this list tokenizes to [UNK], which is fine — the tests only check
# shape and masking behavior, not semantic fidelity.
_VOCAB_TOKENS = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "the", "a", "an", "in", "of", "to", "and", "is", "was", "for", "on",
    "stock", "market", "rose", "sharply", "today", ".",
    "scientists", "discovered", "new", "species", "frog", "amazon",
    "team", "won", "championship", "game", "overtime",
]


@pytest.fixture(scope="module")
def model_and_tokenizer(tmp_path_factory):
    vocab_dir = tmp_path_factory.mktemp("bert_vocab")
    vocab_file = vocab_dir / "vocab.txt"
    vocab_file.write_text("\n".join(_VOCAB_TOKENS) + "\n")

    tokenizer = BertTokenizer(vocab_file=str(vocab_file))

    config = BertConfig(
        vocab_size=len(_VOCAB_TOKENS),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=4,
    )
    model = BertForSequenceClassification(config)
    model.eval()
    return model, tokenizer


# --- transformer.py utilities ---


class TestTokenizeTexts:
    def test_output_shape(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        encoded = tokenize_texts(tokenizer, TEXTS, max_length=MAX_LENGTH)
        assert encoded["input_ids"].shape == (len(TEXTS), MAX_LENGTH)
        assert encoded["attention_mask"].shape == (len(TEXTS), MAX_LENGTH)

    def test_padding_present(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        encoded = tokenize_texts(tokenizer, TEXTS, max_length=MAX_LENGTH)
        # At least some positions should be padding (0 in attention_mask)
        assert (encoded["attention_mask"] == 0).any()


class TestMaskPadding:
    def test_zeros_padding_positions(self):
        attrs = np.ones((2, 5))
        mask = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        result = mask_padding(attrs, mask)
        assert result[0, 3] == 0.0
        assert result[0, 4] == 0.0
        assert result[1, 2] == 0.0
        assert result[0, 0] == 1.0


class TestClassificationTokenIndex:
    def test_encoder_returns_zero(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        encoded = tokenize_texts(tokenizer, TEXTS, max_length=MAX_LENGTH)
        idx = get_classification_token_index(model, encoded["input_ids"], tokenizer)
        assert idx == 0


# --- attention.py ---


class TestExplainAttention:
    def test_output_shape(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        attrs = explain_attention(
            model, tokenizer, TEXTS, max_length=MAX_LENGTH
        )
        assert attrs.shape == (len(TEXTS), MAX_LENGTH)

    def test_non_zero_at_real_tokens(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        attrs = explain_attention(
            model, tokenizer, TEXTS, max_length=MAX_LENGTH
        )
        # Real tokens should have non-zero attributions
        encoded = tokenize_texts(tokenizer, TEXTS, max_length=MAX_LENGTH)
        mask = encoded["attention_mask"].numpy()
        real_token_attrs = attrs[mask == 1]
        assert np.any(real_token_attrs != 0)

    def test_zero_at_padding(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        attrs = explain_attention(
            model, tokenizer, TEXTS, max_length=MAX_LENGTH
        )
        encoded = tokenize_texts(tokenizer, TEXTS, max_length=MAX_LENGTH)
        mask = encoded["attention_mask"].numpy()
        padding_attrs = attrs[mask == 0]
        np.testing.assert_array_equal(padding_attrs, 0.0)

    def test_strategies_produce_different_results(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        a1 = explain_attention(
            model, tokenizer, TEXTS, max_length=MAX_LENGTH, strategy="last_layer_mean"
        )
        a2 = explain_attention(
            model, tokenizer, TEXTS, max_length=MAX_LENGTH, strategy="all_layer_mean"
        )
        # Different strategies should generally produce different values
        assert not np.allclose(a1, a2)

    def test_pipeline_compatible(self, model_and_tokenizer):
        """Verify output feeds into DriftDetector without error."""
        from expl_drift.drift.detector import DriftDetector

        model, tokenizer = model_and_tokenizer
        attrs = explain_attention(
            model, tokenizer, TEXTS, max_length=MAX_LENGTH
        )
        detector = DriftDetector(attrs)
        result = detector.evaluate_window(attrs)
        assert "cosine_drift" in result

    def test_config_attn_implementation_preserved(self, model_and_tokenizer):
        """model.config state around _attn_implementation must be unchanged.

        The function temporarily switches to eager attention to request
        attention outputs, but must fully restore the original config
        value afterwards so that later inference behavior is unaffected.
        This is API-agnostic: it works whether ``_attn_implementation``
        is a plain attribute (older transformers) or a property (newer).
        """
        model, tokenizer = model_and_tokenizer
        had_before = hasattr(model.config, "_attn_implementation")
        value_before = (
            getattr(model.config, "_attn_implementation", None) if had_before else None
        )

        explain_attention(model, tokenizer, TEXTS, max_length=MAX_LENGTH)

        had_after = hasattr(model.config, "_attn_implementation")
        value_after = (
            getattr(model.config, "_attn_implementation", None) if had_after else None
        )

        assert had_before == had_after, (
            "explain_attention changed whether _attn_implementation exists on config"
        )
        assert value_before == value_after, (
            f"explain_attention leaked config change: "
            f"_attn_implementation was {value_before!r}, now {value_after!r}"
        )


# --- ig_transformer.py ---


class TestExplainIGTransformer:
    def test_output_shape(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        attrs = explain_ig_transformer(
            model, tokenizer, TEXTS, max_length=MAX_LENGTH, n_steps=5
        )
        assert attrs.shape == (len(TEXTS), MAX_LENGTH)

    def test_non_zero_at_real_tokens(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        attrs = explain_ig_transformer(
            model, tokenizer, TEXTS, max_length=MAX_LENGTH, n_steps=5
        )
        encoded = tokenize_texts(tokenizer, TEXTS, max_length=MAX_LENGTH)
        mask = encoded["attention_mask"].numpy()
        real_token_attrs = attrs[mask == 1]
        assert np.any(real_token_attrs != 0)

    def test_zero_at_padding(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        attrs = explain_ig_transformer(
            model, tokenizer, TEXTS, max_length=MAX_LENGTH, n_steps=5
        )
        encoded = tokenize_texts(tokenizer, TEXTS, max_length=MAX_LENGTH)
        mask = encoded["attention_mask"].numpy()
        padding_attrs = attrs[mask == 0]
        np.testing.assert_array_equal(padding_attrs, 0.0)

    def test_pipeline_compatible(self, model_and_tokenizer):
        """Verify output feeds into DriftDetector without error."""
        from expl_drift.drift.detector import DriftDetector

        model, tokenizer = model_and_tokenizer
        attrs = explain_ig_transformer(
            model, tokenizer, TEXTS, max_length=MAX_LENGTH, n_steps=5
        )
        detector = DriftDetector(attrs)
        result = detector.evaluate_window(attrs)
        assert "cosine_drift" in result
