# RescueEye Hacker UI

This version keeps the multi-object detection features and adds:

- Neon green mission-control interface
- Dark grid and scanline background
- Radar animation and system status strip
- Terminal-style settings panel
- HUD-style result cards and downloads
- Matching neon overlay on the annotated video
- Green technical styling in the generated PDF report

## Replace an existing project

Replace these files in your current RescueEye folder:

- `app.py`
- `analyzer.py`
- `report.py`

Also copy the `.streamlit` folder.

## Run

Open a terminal inside the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

This direct command avoids the PowerShell activation issue.

## First-time setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
