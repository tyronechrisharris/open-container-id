import logging
import os
import random
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from container_id.config.models import RTSPConfig
from container_id.runtime.consensus import ConsensusEngine
from container_id.runtime.pipeline import RuntimePipeline
from container_id.runtime.tracking import IoUTracker
from container_id.streams.queue import LatestFrameQueue

logger = logging.getLogger(__name__)


def redact_uri(uri: str) -> str:
    """Redact credentials from RTSP URI for logging."""
    if not uri:
        return uri
    # Pattern: rtsp://user:pass@host... -> rtsp://user:***@host...
    return re.sub(r"(rtsp://[^:]+:)[^@]+(@)", r"\1***\2", uri)


class RTSPRunner:
    def __init__(self, config_path: str, models_dir: str):
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f)

        self.config = RTSPConfig(**data)
        self.models_dir = Path(models_dir)
        self.queue = LatestFrameQueue(maxsize=3)
        self.running = False

        self.camera_id = self.config.camera.id
        self.fps = self.config.camera.selected_frame_rate

        uri_env_var = self.config.camera.url_env
        self.uri = os.environ.get(uri_env_var)
        if not self.uri:
            raise ValueError(f"Environment variable {uri_env_var} not set or empty.")

    def _producer_thread(self):
        try:
            import av
        except ImportError:
            logger.error("PyAV ('av' package) is required for RTSP decoding.")
            self.running = False
            return

        reconnect_conf = self.config.camera.reconnect
        delay = reconnect_conf.initial_delay_seconds

        # Options for av.open
        options = {
            "rtsp_transport": self.config.camera.transport,
            "stimeout": str(self.config.camera.connect_timeout_seconds * 1000000),
        }

        while self.running:
            container = None
            try:
                logger.info(f"Connecting to {redact_uri(self.uri)}...")
                container = av.open(self.uri, options=options)
                stream = container.streams.video[0]

                # Reset delay on successful connection
                delay = reconnect_conf.initial_delay_seconds
                logger.info("Connected.")

                # Decoding loop
                if stream.average_rate and stream.average_rate > 0 and self.fps > 0:
                    stream_fps = float(stream.average_rate)
                    if self.fps < stream_fps:
                        # calculate how many frames to skip approximately
                        pass  # PyAV timestamps are better.

                last_processed_time = time.monotonic()
                target_interval = 1.0 / self.fps if self.fps > 0 else 0

                for frame in container.decode(stream):
                    if not self.running:
                        break

                    now = time.monotonic()
                    if target_interval > 0 and (now - last_processed_time) < target_interval:
                        continue

                    last_processed_time = now

                    # Convert to numpy (BGR for cv2 compatibility used elsewhere)
                    img = frame.to_ndarray(format="bgr24")

                    # PTS to real timestamp approximation
                    utc_now = datetime.now(UTC)

                    self.queue.put({"image": img, "timestamp": utc_now})

            except Exception as e:  # noqa: BLE001
                if self.running:
                    logger.error(
                        f"Decoder error or disconnected: {e}. Reconnecting in {delay} seconds..."
                    )
                    time.sleep(delay)
                    # Exponential backoff with jitter
                    jitter = (
                        delay * reconnect_conf.jitter_fraction * random.uniform(-1, 1)
                    )
                    delay = min(
                        reconnect_conf.maximum_delay_seconds,
                        delay * reconnect_conf.multiplier + jitter,
                    )
            finally:
                if container:
                    import contextlib

                    with contextlib.suppress(Exception):
                        container.close()

    def _consumer_thread(self):
        pipeline = RuntimePipeline(str(self.models_dir))
        tracker = IoUTracker(iou_threshold=0.30, max_missed_frames=10)
        consensus = ConsensusEngine(camera_id=self.camera_id)

        output_file = Path("events.local.jsonl")

        # Write mode append or w? The spec says output to events.local.jsonl or similar
        with open(output_file, "a") as out_f:
            while self.running:
                item = self.queue.get(timeout=1.0)
                if not item:
                    continue

                img = item["image"]
                ts = item["timestamp"]

                try:
                    res = pipeline.process_frame(img)
                    tracks = tracker.update(res.get("detections", []), ts)

                    for track in tracks:
                        event = consensus.process_track(track, ts)
                        if event:
                            logger.info(f"Event emitted: {event.container_number}")
                            out_f.write(event.model_dump_json() + "\n")
                            out_f.flush()
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Consumer inference error: {e}")

    def run(self):
        logger.info(f"Starting RTSP Runner for camera {self.camera_id}")
        self.running = True

        prod = threading.Thread(target=self._producer_thread, daemon=True)
        cons = threading.Thread(target=self._consumer_thread, daemon=True)

        prod.start()
        cons.start()

        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping RTSP Runner...")
            self.running = False
            prod.join(timeout=2.0)
            cons.join(timeout=2.0)
