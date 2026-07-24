from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from analyzer import AnalysisResult


NEON = colors.HexColor("#29C768")
DARK = colors.HexColor("#07140C")
GRID = colors.HexColor("#A9C8B2")
PALE = colors.HexColor("#EAF8ED")


def build_pdf_report(result: AnalysisResult, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    styles = getSampleStyleSheet()
    styles["Title"].textColor = DARK
    styles["Heading2"].textColor = colors.HexColor("#11763D")

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"RescueEye Report - {result.mission_name}",
    )

    story = [
        Paragraph("RESCUE EYE MISSION REPORT", styles["Title"]),
        Spacer(1, 8),
        Paragraph(
            "AI-assisted review of recorded drone footage for selected common objects.",
            styles["Normal"],
        ),
        Spacer(1, 14),
    ]

    mission_data = [
        ["Mission", result.mission_name],
        ["Location", result.location or "Not provided"],
        ["Operator", result.operator or "Not provided"],
        ["Video duration", f"{result.duration_seconds:.2f} seconds"],
        ["Processed frames", str(result.processed_frames)],
        ["Estimated tracked objects", str(result.unique_track_ids)],
        ["Frames containing detections", str(result.total_detection_frames)],
        ["Person low-movement alerts", str(result.low_movement_alerts)],
        ["Selected classes", ", ".join(result.selected_classes)],
    ]

    mission_table = Table(mission_data, colWidths=[55 * mm, 110 * mm])
    mission_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), PALE),
                ("TEXTCOLOR", (0, 0), (0, -1), DARK),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 7),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(mission_table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("OBJECT SUMMARY", styles["Heading2"]))
    story.append(Spacer(1, 6))

    if result.class_counts:
        class_rows = [["Object class", "Estimated tracked objects"]]
        for class_name, count in result.class_counts.items():
            class_rows.append([class_name.title(), str(count)])

        class_table = Table(class_rows, repeatRows=1, colWidths=[95 * mm, 60 * mm])
        class_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), DARK),
                    ("TEXTCOLOR", (0, 0), (-1, 0), NEON),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(class_table)
    else:
        story.append(Paragraph("No selected objects were detected.", styles["Normal"]))

    story.append(Spacer(1, 18))
    story.append(Paragraph("DETECTION EVENTS", styles["Heading2"]))
    story.append(Spacer(1, 6))

    if not result.events:
        story.append(Paragraph("No detection events were recorded.", styles["Normal"]))
    else:
        event_rows = [["Type", "Object", "Track", "Time", "Zone", "Confidence"]]
        for event in result.events:
            event_rows.append(
                [
                    event.event_type,
                    event.object_class.title(),
                    str(event.track_id),
                    f"{event.timestamp_seconds:.2f}s",
                    event.zone,
                    f"{event.confidence:.2f}",
                ]
            )

        event_table = Table(
            event_rows,
            repeatRows=1,
            colWidths=[48 * mm, 28 * mm, 18 * mm, 22 * mm, 18 * mm, 27 * mm],
        )
        event_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), DARK),
                    ("TEXTCOLOR", (0, 0), (-1, 0), NEON),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(event_table)

    story.append(PageBreak())
    story.append(Paragraph("IMPORTANT LIMITATIONS", styles["Heading2"]))
    story.append(Spacer(1, 6))

    limitations = [
        "The standard pretrained model recognizes common COCO objects. It does not reliably detect fire, smoke, floodwater, injuries, or structural damage.",
        "Small, distant, partly hidden, blurred, or low-contrast objects may be missed.",
        "A tracking ID is not a confirmed object count. IDs can change if tracking is lost.",
        "Low apparent movement is calculated only for people and is not a medical or injury assessment.",
        "Parked vehicles, unattended bags, animals, and other detections do not automatically indicate danger.",
        "All alerts require human review. This prototype must not be used as the only search method in a real emergency.",
    ]

    for item in limitations:
        story.append(Paragraph(f"• {item}", styles["Normal"]))
        story.append(Spacer(1, 5))

    document.build(story)
    return output_path
