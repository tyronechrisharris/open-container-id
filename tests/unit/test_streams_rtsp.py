from unittest.mock import patch

from container_id.streams.rtsp import RTSPRunner


@patch("container_id.streams.rtsp.os.environ.get")
def test_rtsp_runner_init(mock_env_get, tmp_path):
    mock_env_get.return_value = "rtsp://test"
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
camera:
  id: "cam1"
  url_env: "RTSP_URL"
  selected_frame_rate: 15
""")
    models_dir = tmp_path / "models"

    with patch("container_id.streams.rtsp.RuntimePipeline"):
        runner = RTSPRunner(str(config_file), str(models_dir))
        assert runner.fps == 15
