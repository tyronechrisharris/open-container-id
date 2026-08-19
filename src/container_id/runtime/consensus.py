import uuid
from collections import defaultdict
from datetime import datetime

from container_id.runtime.interfaces import ContainerEvent, OCRCandidate
from container_id.runtime.tracking import Track


class DuplicateSuppressor:
    def __init__(self, suppression_seconds: float = 30.0):
        self.suppression_seconds = suppression_seconds
        # Dict of (camera_id, container_number) -> last_emitted_timestamp
        self.history: dict[tuple[str, str], datetime] = {}

    def should_suppress(
        self, camera_id: str, container_number: str, current_time: datetime
    ) -> bool:
        key = (camera_id, container_number)
        last_emitted = self.history.get(key)

        if last_emitted is None:
            return False

        elapsed = (current_time - last_emitted).total_seconds()
        return elapsed < self.suppression_seconds

    def record_emission(
        self, camera_id: str, container_number: str, current_time: datetime
    ) -> None:
        self.history[(camera_id, container_number)] = current_time

    def clean_history(self, current_time: datetime) -> None:
        """Removes expired entries from history to prevent memory unbounded growth."""
        keys_to_remove = []
        for k, v in self.history.items():
            if (current_time - v).total_seconds() > self.suppression_seconds:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self.history[k]


class ConsensusEngine:
    def __init__(
        self,
        minimum_supporting_frames: int = 3,
        window_frames: int = 7,
        minimum_weighted_score: float = 2.2,
        duplicate_suppression_seconds: float = 30.0,
        require_iso_structure: bool = True,
        require_valid_check_digit: bool = True,
        camera_id: str = "default_camera",
        model_bundle_version: str = "unknown",
    ):
        self.minimum_supporting_frames = minimum_supporting_frames
        self.window_frames = window_frames
        self.minimum_weighted_score = minimum_weighted_score
        self.require_iso_structure = require_iso_structure
        self.require_valid_check_digit = require_valid_check_digit
        self.camera_id = camera_id
        self.model_bundle_version = model_bundle_version

        self.suppressor = DuplicateSuppressor(duplicate_suppression_seconds)

    def _calculate_candidate_weight(self, track: Track, candidate_dict: dict) -> float:
        """
        Calculate the weight of a specific frame candidate.
        candidate_dict looks like:
        { "timestamp": dt, "candidate": { "normalized_text": "...", "confidence": 0.9, "transform": "identity", ...} }
        We also need the crop_quality and detector_confidence for this frame.
        Since track lists are appended in sync, we assume index alignment or we can approximate.
        For exactness, we will assume candidates and track metrics align.
        """
        # We need the index of this candidate in the track to look up corresponding detector/crop scores.
        try:
            idx = track.ocr_candidates.index(candidate_dict)
            det_conf = track.detector_confidences[idx]
            crop_qual = track.crop_quality_scores[idx]
        except (ValueError, IndexError):
            # Fallback if alignment is broken
            det_conf = 0.5
            crop_qual = 0.5

        cand_info = candidate_dict["candidate"]
        if isinstance(cand_info, dict):
            ocr_conf = cand_info.get("confidence", 0.0)
            transform = cand_info.get("transform", "identity")
            corrections = cand_info.get("corrections", [])
        else:
            ocr_conf = cand_info.confidence
            transform = cand_info.transform
            corrections = cand_info.corrections

        # Factor penalties
        transform_factor = 1.0 if transform == "identity" else 0.9
        correction_factor = 1.0 if not corrections else 0.8

        return det_conf * ocr_conf * crop_qual * transform_factor * correction_factor

    def process_track(
        self, track: Track, current_timestamp: datetime
    ) -> ContainerEvent | None:
        if track.emitted_event:
            return None

        # Clean history periodically (could do this less frequently in a real tight loop, but fine for MVP)
        self.suppressor.clean_history(current_timestamp)

        # Get recent window of candidates
        recent_candidates = track.ocr_candidates[-self.window_frames :]
        if not recent_candidates:
            return None

        # Group by normalized text
        scores_by_text: defaultdict[str, float] = defaultdict(float)
        frames_by_text: defaultdict[str, int] = defaultdict(int)
        best_candidate_obj_by_text: dict[str, OCRCandidate] = {}
        max_ocr_conf_by_text: defaultdict[str, float] = defaultdict(float)

        for c_dict in recent_candidates:
            c = c_dict["candidate"]
            is_dict = isinstance(c, dict)

            norm_text = c.get("normalized_text") if is_dict else c.normalized_text
            if not norm_text:
                continue

            struct_valid = (
                c.get("structure_valid", False) if is_dict else c.structure_valid
            )
            check_valid = (
                c.get("check_digit_valid", False) if is_dict else c.check_digit_valid
            )

            if self.require_iso_structure and not struct_valid:
                continue
            if self.require_valid_check_digit and not check_valid:
                continue

            weight = self._calculate_candidate_weight(track, c_dict)
            scores_by_text[norm_text] += weight
            frames_by_text[norm_text] += 1

            ocr_conf = c.get("confidence", 0.0) if is_dict else c.confidence
            if ocr_conf > max_ocr_conf_by_text[norm_text]:
                max_ocr_conf_by_text[norm_text] = ocr_conf
                best_candidate_obj_by_text[norm_text] = c_dict

        if not scores_by_text:
            return None

        # Find the best candidate
        best_text = max(scores_by_text.keys(), key=lambda k: scores_by_text[k])
        best_score = scores_by_text[best_text]
        support_frames = frames_by_text[best_text]

        # Check thresholds
        if support_frames < self.minimum_supporting_frames:
            return None
        if best_score < self.minimum_weighted_score:
            return None

        # Check duplicate suppression
        if self.suppressor.should_suppress(
            self.camera_id, best_text, current_timestamp
        ):
            # Optionally record that we hit suppression, but we don't emit
            return None

        # Mark emitted
        track.emitted_event = True
        track.current_consensus = best_text
        self.suppressor.record_emission(self.camera_id, best_text, current_timestamp)

        # Build event
        best_c = best_candidate_obj_by_text[best_text]
        if isinstance(best_c, dict):
            best_c = best_c["candidate"]
        is_dict = isinstance(best_c, dict)

        # Calculate aggregates
        avg_det_conf = sum(track.detector_confidences) / len(track.detector_confidences)

        if is_dict:
            check_digit_valid = best_c.get("check_digit_valid", False) # type: ignore
        else:
            check_digit_valid = best_c.check_digit_valid # type: ignore

        event = ContainerEvent(
            event_id=str(uuid.uuid4()),
            timestamp_utc=current_timestamp,
            camera_id=self.camera_id,
            track_id=track.track_id,
            container_number=best_text,
            check_digit_valid=check_digit_valid,
            confidence=float(
                min(1.0, best_score / (self.window_frames * 1.0))
            ),  # rough normalization
            detector_confidence=avg_det_conf,
            ocr_confidence=max_ocr_conf_by_text[best_text],
            supporting_frames=support_frames,
            first_seen_utc=track.first_timestamp,
            confirmed_at_utc=current_timestamp,
            bbox_xyxy=track.last_box,
            model_bundle_version=self.model_bundle_version,
        )

        return event
