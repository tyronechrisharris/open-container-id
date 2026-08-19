from datetime import UTC, datetime, timedelta

from container_id.runtime.tracking import IoUTracker, compute_iou


def test_compute_iou():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (5.0, 5.0, 15.0, 15.0)

    iou = compute_iou(box1, box2)
    # intersection: 5x5 = 25
    # union: 100 + 100 - 25 = 175
    # iou = 25 / 175 = 1/7 = 0.1428...
    assert round(iou, 4) == 0.1429

    # disjoint
    box3 = (20.0, 20.0, 30.0, 30.0)
    assert compute_iou(box1, box3) == 0.0


def test_iou_tracker_match_and_age():
    tracker = IoUTracker(iou_threshold=0.3, max_missed_frames=2)
    t0 = datetime.now(UTC)

    # Frame 1: One detection
    dets1 = [{"bbox_xyxy": (0.0, 0.0, 10.0, 10.0), "detector_confidence": 0.9}]
    tracks1 = tracker.update(dets1, t0)
    assert len(tracks1) == 1
    assert tracks1[0].missed_frames == 0
    track_id = tracks1[0].track_id

    # Frame 2: Slight movement (should match)
    t1 = t0 + timedelta(seconds=1)
    dets2 = [{"bbox_xyxy": (1.0, 1.0, 11.0, 11.0), "detector_confidence": 0.85}]
    tracks2 = tracker.update(dets2, t1)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == track_id
    assert tracks2[0].missed_frames == 0
    assert len(tracks2[0].detector_confidences) == 2

    # Frame 3: Disjoint detection (should create new track, age old one)
    t2 = t1 + timedelta(seconds=1)
    dets3 = [{"bbox_xyxy": (50.0, 50.0, 60.0, 60.0)}]
    tracks3 = tracker.update(dets3, t2)
    assert len(tracks3) == 2

    # Find original track
    orig = next(t for t in tracks3 if t.track_id == track_id)
    assert orig.missed_frames == 1

    # Frame 4: Empty detection (ages both, keeps them alive because max_missed_frames=2)
    t3 = t2 + timedelta(seconds=1)
    tracks4 = tracker.update([], t3)
    assert len(tracks4) == 2

    # Frame 5: Empty detection (ages both, kills original track because it hits missed_frames = 3 > 2)
    t4 = t3 + timedelta(seconds=1)
    tracks5 = tracker.update([], t4)
    assert len(tracks5) == 1
    assert tracks5[0].track_id != track_id
