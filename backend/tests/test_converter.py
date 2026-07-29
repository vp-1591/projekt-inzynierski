import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from app.training.converter import _strip_bnb_config, main

# ---------------------------------------------------------------------------
# _strip_bnb_config tests
# ---------------------------------------------------------------------------


def test_strip_bnb_config_removes_quantization_config():
    """A config.json that contains quantization_config should be rewritten
    without that key, and the returned directory should be a new temp dir."""
    with tempfile.TemporaryDirectory() as base_dir:
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
            },
        }
        config_path = os.path.join(base_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        result = _strip_bnb_config(base_dir)

        try:
            assert result != base_dir
            clean_config_path = os.path.join(result, "config.json")
            with open(clean_config_path, encoding="utf-8") as f:
                clean_config = json.load(f)

            assert "quantization_config" not in clean_config
            assert clean_config["architectures"] == ["LlamaForCausalLM"]
            assert clean_config["model_type"] == "llama"
        finally:
            # Clean up the temp dir created by _strip_bnb_config
            if result != base_dir:
                import shutil

                shutil.rmtree(result, ignore_errors=True)


def test_strip_bnb_config_noop_when_no_quantization_config():
    """If config.json exists but has no quantization_config, the original
    directory should be returned unchanged."""
    with tempfile.TemporaryDirectory() as base_dir:
        config = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
        config_path = os.path.join(base_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        result = _strip_bnb_config(base_dir)

        assert result == base_dir


def test_strip_bnb_config_handles_missing_config_json():
    """If no config.json exists in base_dir, return base_dir unchanged."""
    with tempfile.TemporaryDirectory() as base_dir:
        # No config.json created
        result = _strip_bnb_config(base_dir)

        assert result == base_dir


def test_strip_bnb_config_preserves_other_keys():
    """All keys other than quantization_config should be preserved in the
    cleaned config.json."""
    with tempfile.TemporaryDirectory() as base_dir:
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "quantization_config": {"load_in_4bit": True},
            "vocab_size": 32000,
        }
        config_path = os.path.join(base_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        result = _strip_bnb_config(base_dir)

        try:
            clean_config_path = os.path.join(result, "config.json")
            with open(clean_config_path, encoding="utf-8") as f:
                clean_config = json.load(f)

            # quantization_config removed
            assert "quantization_config" not in clean_config
            # All other keys preserved exactly
            for key in ("architectures", "model_type", "hidden_size", "num_attention_heads", "vocab_size"):
                assert key in clean_config
                assert clean_config[key] == config[key]
        finally:
            if result != base_dir:
                import shutil

                shutil.rmtree(result, ignore_errors=True)


def test_strip_bnb_config_creates_temp_dir_with_prefix_gguf_base():
    """The temp directory created by _strip_bnb_config should use the
    prefix 'gguf_base_'."""
    with tempfile.TemporaryDirectory() as base_dir:
        config = {"quantization_config": {"load_in_4bit": True}}
        config_path = os.path.join(base_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        result = _strip_bnb_config(base_dir)

        try:
            assert result != base_dir
            dirname = os.path.basename(result)
            assert dirname.startswith("gguf_base_")
        finally:
            if result != base_dir:
                import shutil

                shutil.rmtree(result, ignore_errors=True)


@pytest.mark.skipif(os.name == "nt", reason="os.symlink requires privileges on Windows")
def test_strip_bnb_config_symlinks_non_config_files():
    """Non-config.json files in base_dir should be symlinked (not copied)
    into the temp directory."""
    with tempfile.TemporaryDirectory() as base_dir:
        config = {"quantization_config": {"load_in_4bit": True}}
        config_path = os.path.join(base_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        # Create a non-config file
        weights_path = os.path.join(base_dir, "model.safetensors")
        with open(weights_path, "w") as f:
            f.write("fake weights")

        result = _strip_bnb_config(base_dir)

        try:
            linked = os.path.join(result, "model.safetensors")
            assert os.path.islink(linked)
        finally:
            if result != base_dir:
                import shutil

                shutil.rmtree(result, ignore_errors=True)


# ---------------------------------------------------------------------------
# main() CLI tests
# ---------------------------------------------------------------------------


def test_main_requires_base_or_base_model_id():
    """If neither --base nor --base-model-id is provided, main() should
    call sys.exit(1)."""
    with patch("sys.argv", ["converter", "--adapter", "/path/to/adapter", "--output", "/out/model.gguf"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_main_accepts_base_model_id():
    """When --base-model-id is provided, the conversion command should include
    --base-model-id and _strip_bnb_config should not be called."""
    with (
        patch(
            "sys.argv",
            [
                "converter",
                "--adapter",
                "/path/to/adapter",
                "--output",
                "/out/model.gguf",
                "--base-model-id",
                "speakleash/Bielik-4.5B-v3",
            ],
        ),
        patch("app.training.converter.subprocess") as mock_subprocess,
        patch("app.training.converter._strip_bnb_config") as mock_strip,
        patch("os.path.exists", return_value=True),
    ):
        mock_subprocess.check_call = MagicMock()
        main()

        # _strip_bnb_config should NOT be called when using --base-model-id
        mock_strip.assert_not_called()

        # The command should include --base-model-id
        call_args = mock_subprocess.check_call.call_args[0][0]
        assert "--base-model-id" in call_args
        assert "speakleash/Bielik-4.5B-v3" in call_args


def test_main_accepts_base_path():
    """When --base is provided with a local path, _strip_bnb_config should be
    called and the command should include --base with the (possibly cleaned)
    directory."""
    with (
        patch(
            "sys.argv",
            [
                "converter",
                "--adapter",
                "/path/to/adapter",
                "--output",
                "/out/model.gguf",
                "--base",
                "/path/to/base",
            ],
        ),
        patch("app.training.converter.subprocess") as mock_subprocess,
        patch("app.training.converter._strip_bnb_config", return_value="/cleaned/base") as mock_strip,
        patch("os.path.exists", return_value=True),
        patch("shutil.rmtree"),
    ):
        mock_subprocess.check_call = MagicMock()
        main()

        # _strip_bnb_config should be called with the base path
        mock_strip.assert_called_once_with("/path/to/base")

        # The command should include --base with the cleaned dir
        call_args = mock_subprocess.check_call.call_args[0][0]
        assert "--base" in call_args
        assert "/cleaned/base" in call_args


def test_main_cleans_up_temp_dir_on_success():
    """When _strip_bnb_config returns a temp dir (different from base),
    shutil.rmtree should be called on it after successful conversion."""
    with (
        patch(
            "sys.argv",
            [
                "converter",
                "--adapter",
                "/path/to/adapter",
                "--output",
                "/out/model.gguf",
                "--base",
                "/path/to/base",
            ],
        ),
        patch("app.training.converter.subprocess") as mock_subprocess,
        patch("app.training.converter._strip_bnb_config", return_value="/tmp/gguf_base_xyz"),
        patch("os.path.exists", return_value=True),
        patch("app.training.converter.shutil") as mock_shutil,
    ):
        mock_subprocess.check_call = MagicMock()
        main()

        # Cleanup should be called on the temp dir
        mock_shutil.rmtree.assert_called_once_with("/tmp/gguf_base_xyz", ignore_errors=True)


def test_main_no_cleanup_when_base_unchanged():
    """When _strip_bnb_config returns the original dir (no quantization_config
    to strip), shutil.rmtree should NOT be called on it."""
    with (
        patch(
            "sys.argv",
            [
                "converter",
                "--adapter",
                "/path/to/adapter",
                "--output",
                "/out/model.gguf",
                "--base",
                "/path/to/base",
            ],
        ),
        patch("app.training.converter.subprocess") as mock_subprocess,
        patch("app.training.converter._strip_bnb_config", return_value="/path/to/base"),
        patch("os.path.exists", return_value=True),
        patch("app.training.converter.shutil") as mock_shutil,
    ):
        mock_subprocess.check_call = MagicMock()
        main()

        # No cleanup since the dir was not changed
        mock_shutil.rmtree.assert_not_called()
