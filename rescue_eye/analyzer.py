from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
from ultralytics import YOLO


@dataclass
class DetectionEvent:
    event_type: str
    object_class: str
    track_id: int
    timestamp_seconds: float
    confidence: float
    zone: str
    evidence_path: str
    note: str


@dataclass
class AnalysisResult:
    mission_name: str
    location: str
    operator: str
    source_video: str
    annotated_video: str
    events_csv: str
    duration_seconds: float
    processed_frames: int
    total_frames: int
    unique_track_ids: int
    total_detection_frames: int
    low_movement_alerts: int
    selected_classes: list[str]
    class_counts: dict[str, int]
    events: list[DetectionEvent]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["events"] = [asdict(event) for event in self.events]
        return data


class DroneVideoAnalyzer:
    """Review drone footage for selected common objects."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.35,
        frame_step: int = 2,
        low_movement_seconds: float = 2.0,
        movement_threshold_ratio: float = 0.025,
        selected_class_names: tuple[str, ...] | None = None,
    ) -> None:
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.frame_step = max(1, frame_step)
        self.low_movement_seconds = max(0.5, low_movement_seconds)
        self.movement_threshold_ratio = movement_threshold_ratio

        model_names = self.model.names
        if isinstance(model_names, dict):
            self.class_names = {int(key): str(value) for key, value in model_names.items()}
        else:
            self.class_names = {index: str(value) for index, value in enumerate(model_names)}

        normalized_selection = {
            name.strip().lower() for name in (selected_class_names or ()) if name.strip()
        }

        if normalized_selection:
            self.class_ids = [
                class_id
                for class_id, class_name in self.class_names.items()
                if class_name.lower() in normalized_selection
            ]
        else:
            self.class_ids = list(self.class_names.keys())


        if not self.class_isds:
            raise ValueError("None of the selected object classes exist in this model.")

        self.selected_class_names = [self.class_names[class_id] for class_id in self.class_ids]

    @staticmethod
    def _screen_zone(cx: float, cy: float, width: int, height: int) -> str:
        columns = ["A", "B", "C"]
        rows = ["1", "2", "3"]
        column_index = min(2, int(cx / max(width, 1) * 3))
        row_index = min(2, int(cy / max(height, 1) * 3))
        return f"{columns[column_index]}{rows[row_index]}"

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"

    @staticmethod
    def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    def _has_low_apparent_movement(
        self,
        history: deque[tuple[float, float, float]],
        frame_diagonal: float,
    ) -> bool:
        if len(history) < 2:
            return False

        elapsed = history[-1][2] - history[0][2]
        if elapsed < self.low_movement_seconds:
            return False

        start = (history[0][0], history[0][1])
        end = (history[-1][0], history[-1][1])
        movement = self._distance(start, end)
        threshold_pixels = frame_diagonal * self.movement_threshold_ratio
        return movement <= threshold_pixels

    @staticmethod
    def _safe_filename_part(value: str) -> str:
        cleaned = "".join(character if character.isalnum() else "_" for character in value)
        return cleaned.strip("_") or "object"

    @classmethod
    def _save_evidence(
        cls,
        frame,
        output_folder: Path,
        event_type: str,
        object_class: str,
        track_id: int,
        timestamp_seconds: float,
    ) -> Path:
        evidence_folder = output_folder / "evidence"
        evidence_folder.mkdir(parents=True, exist_ok=True)

        safe_time = f"{timestamp_seconds:.2f}".replace(".", "_")
        safe_event = cls._safe_filename_part(event_type)
        safe_class = cls._safe_filename_part(object_class)
        filename = f"{safe_event}_{safe_class}_track_{track_id}_{safe_time}s.jpg"
        path = evidence_folder / filename
        cv2.imwrite(str(path), frame)
        return path

    @staticmethod
    def _write_events_csv(events: list[DetectionEvent], csv_path: Path) -> None:
        columns = [
            "event_type",
            "object_class",
            "track_id",
            "timestamp_seconds",
            "confidence",
            "zone",
            "evidence_path",
            "note",
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for event in events:
                writer.writerow(asdict(event))

    @staticmethod
    def _box_color(object_class: str, low_movement: bool) -> tuple[int, int, int]:
        if low_movement:
            return (45, 70, 255)
        if object_class == "person":
            return (95, 255, 70)
        if object_class in {
            "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"
        }:
            return (255, 235, 70)
        if object_class == "boat":
            return (255, 255, 65)
        if object_class in {
            "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe",
        }:
            return (80, 215, 255)
        if object_class in {"backpack", "umbrella", "handbag", "suitcase"}:
            return (255, 90, 245)
        return (130, 255, 145)

    @staticmethod
    def _draw_label(frame, label: str, x1: int, y1: int, color, frame_width: int) -> None:
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.48
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_width = min(text_width + 14, max(10, frame_width - x1))
        top = max(0, y1 - text_height - 16)

        cv2.rectangle(frame, (x1, top), (x1 + label_width, y1), (2, 12, 6), -1)
        cv2.rectangle(frame, (x1, top), (x1 + label_width, y1), color, 1)
        cv2.putText(
            frame,
            label,
            (x1 + 6, max(text_height + 2, y1 - 7)),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_corner_box(frame, x1: int, y1: int, x2: int, y2: int, color) -> None:
        length = max(8, min(24, (x2 - x1) // 4, (y2 - y1) // 4))
        thickness = 2

        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

    @staticmethod
    def _draw_hud(
        frame,
        mission_name: str,
        timestamp: str,
        active_tracks: int,
        width: int,
        height: int,
    ) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 72), (2, 10, 5), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        green = (90, 255, 75)
        cyan = (255, 235, 70)
        dim_green = (45, 105, 55)

        cv2.putText(
            frame, "RESCUE//EYE", (16, 25), cv2.FONT_HERSHEY_DUPLEX,
            0.58, green, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"MISSION: {mission_name[:32].upper()}", (16, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (205, 235, 210), 1, cv2.LINE_AA,
        )

        right_text = f"T+ {timestamp}  //  TRACKS {active_tracks:02d}"
        (right_width, _), _ = cv2.getTextSize(
            right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        cv2.putText(
            frame, right_text, (max(16, width - right_width - 16), 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, cyan, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "CV LINK: ACTIVE  //  GRID: 3X3", (max(16, width - 230), 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, green, 1, cv2.LINE_AA,
        )

        for x in (width // 3, 2 * width // 3):
            cv2.line(frame, (x, 72), (x, height), dim_green, 1)
        for y in (72 + (height - 72) // 3, 72 + 2 * (height - 72) // 3):
            cv2.line(frame, (0, y), (width, y), dim_green, 1)

        center_x = width // 2
        center_y = (height + 72) // 2
        cv2.line(frame, (center_x - 12, center_y), (center_x - 4, center_y), green, 1)
        cv2.line(frame, (center_x + 4, center_y), (center_x + 12, center_y), green, 1)
        cv2.line(frame, (center_x, center_y - 12), (center_x, center_y - 4), green, 1)
        cv2.line(frame, (center_x, center_y + 4), (center_x, center_y + 12), green, 1)

        bracket = 26
        margin = 10
        cv2.line(frame, (margin, 82), (margin + bracket, 82), green, 2)
        cv2.line(frame, (margin, 82), (margin, 82 + bracket), green, 2)
        cv2.line(
            frame,
            (width - margin, height - margin),
            (width - margin - bracket, height - margin),
            cyan,
            2,
        )
        cv2.line(
            frame,
            (width - margin, height - margin),
            (width - margin, height - margin - bracket),
            cyan,
            2,
        )

    def analyze(
        self,
        video_path: str | Path,
        output_folder: str | Path,
        mission_name: str,
        location: str,
        operator: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        preview_callback: Optional[Callable[[object], None]] = None,
    ) -> AnalysisResult:
        video_path = Path(video_path)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("The uploaded video could not be opened.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if total_frames > 0 else 0.0
        frame_diagonal = math.hypot(width, height)

        annotated_video_path = output_folder / "annotated_mission.mp4"
        writer = cv2.VideoWriter(
            str(annotated_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, fps / self.frame_step),
            (width, height),
        )

        if not writer.isOpened():
            capture.release()
            raise RuntimeError("The annotated output video could not be created.")

        self.model.predictor = None

        track_histories: dict[tuple[str, int], deque[tuple[float, float, float]]] = defaultdict(deque)
        seen_track_keys: set[tuple[str, int]] = set()
        low_movement_keys: set[tuple[str, int]] = set()
        unique_tracks_by_class: dict[str, set[int]] = defaultdict(set)
        events: list[DetectionEvent] = []

        frame_index = 0
        processed_frames = 0
        total_detection_frames = 0
        preview_every = max(1, int((fps / self.frame_step) * 0.5))

        try:
            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    break

                if frame_index % self.frame_step != 0:
                    frame_index += 1
                    continue

                timestamp_seconds = frame_index / fps
                processed_frames += 1

                result = self.model.track(
                    frame,
                    persist=True,
                    classes=self.class_ids,
                    conf=self.confidence,
                    iou=0.5,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )[0]

                object_seen_in_frame = False
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    coordinates = boxes.xyxy.cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    class_ids = boxes.cls.int().cpu().tolist()
                    track_ids = (
                        boxes.id.int().cpu().tolist()
                        if boxes.id is not None
                        else [-1 - index for index in range(len(coordinates))]
                    )

                    for box, confidence, class_id, track_id in zip(
                        coordinates, confidences, class_ids, track_ids
                    ):
                        object_seen_in_frame = True
                        class_id = int(class_id)
                        track_id = int(track_id)
                        object_class = self.class_names.get(class_id, f"class_{class_id}")
                        track_key = (object_class, track_id)
                        unique_tracks_by_class[object_class].add(track_id)

                        x1, y1, x2, y2 = [int(value) for value in box]
                        x1 = max(0, min(width - 1, x1))
                        y1 = max(73, min(height - 1, y1))
                        x2 = max(0, min(width - 1, x2))
                        y2 = max(73, min(height - 1, y2))
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        zone = self._screen_zone(cx, cy, width, height)

                        low_movement = False
                        if object_class == "person":
                            history = track_histories[track_key]
                            history.append((cx, cy, timestamp_seconds))
                            while (
                                history
                                and timestamp_seconds - history[0][2]
                                > self.low_movement_seconds + 1.0
                            ):
                                history.popleft()
                            low_movement = self._has_low_apparent_movement(
                                history, frame_diagonal
                            )

                        if track_key not in seen_track_keys:
                            seen_track_keys.add(track_key)
                            evidence_path = self._save_evidence(
                                frame,
                                output_folder,
                                "first_detection",
                                object_class,
                                track_id,
                                timestamp_seconds,
                            )
                            events.append(
                                DetectionEvent(
                                    event_type="First detection",
                                    object_class=object_class,
                                    track_id=track_id,
                                    timestamp_seconds=round(timestamp_seconds, 2),
                                    confidence=round(float(confidence), 3),
                                    zone=zone,
                                    evidence_path=str(evidence_path),
                                    note=f"A new tracked {object_class} appeared in the footage.",
                                )
                            )

                        if low_movement and track_key not in low_movement_keys:
                            low_movement_keys.add(track_key)
                            evidence_path = self._save_evidence(
                                frame,
                                output_folder,
                                "low_movement",
                                object_class,
                                track_id,
                                timestamp_seconds,
                            )
                            events.append(
                                DetectionEvent(
                                    event_type="Low apparent movement",
                                    object_class=object_class,
                                    track_id=track_id,
                                    timestamp_seconds=round(timestamp_seconds, 2),
                                    confidence=round(float(confidence), 3),
                                    zone=zone,
                                    evidence_path=str(evidence_path),
                                    note=(
                                        "Little person movement was visible relative to the frame. "
                                        "This is not a medical or injury assessment."
                                    ),
                                )
                            )

                        color = self._box_color(object_class, low_movement)
                        status = "LOW MOVEMENT" if low_movement else object_class.upper()
                        label = f"{status} // ID {track_id} // {confidence:.2f} // {zone}"
                        self._draw_corner_box(frame, x1, y1, x2, y2, color)
                        self._draw_label(frame, label, x1, y1, color, width)

                if object_seen_in_frame:
                    total_detection_frames += 1

                self._draw_hud(
                    frame,
                    mission_name,
                    self._format_time(timestamp_seconds),
                    len(seen_track_keys),
                    width,
                    height,
                )

                writer.write(frame)

                if preview_callback and processed_frames % preview_every == 0:
                    preview_callback(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if progress_callback:
                    progress = (
                        min(1.0, frame_index / total_frames)
                        if total_frames > 0
                        else 0.0
                    )
                    progress_callback(
                        progress,
                        f"Analyzing {self._format_time(timestamp_seconds)} of "
                        f"{self._format_time(duration_seconds)}",
                    )

                frame_index += 1
        finally:
            capture.release()
            writer.release()

        events_csv_path = output_folder / "mission_events.csv"
        self._write_events_csv(events, events_csv_path)

        if progress_callback:
            progress_callback(1.0, "Analysis complete")

        class_counts = {
            class_name: len(track_ids)
            for class_name, track_ids in sorted(unique_tracks_by_class.items())
        }

        return AnalysisResult(
            mission_name=mission_name,
            location=location,
            operator=operator,
            source_video=str(video_path),
            annotated_video=str(annotated_video_path),
            events_csv=str(events_csv_path),
            duration_seconds=round(duration_seconds, 2),
            processed_frames=processed_frames,
            total_frames=total_frames,
            unique_track_ids=sum(class_counts.values()),
            total_detection_frames=total_detection_frames,
            low_movement_alerts=len(low_movement_keys),
            selected_classes=list(self.selected_class_names),
            class_counts=class_counts,
            events=events,
        )
