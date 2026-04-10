# Exam Proctor V2

A modular exam monitoring application with these violation categories:

- Looking away from screen (head pose + eye direction)
- Multiple face detection
- Tab/app switching pattern via screen-scene change analysis
- Phone or unauthorized device detected
- Candidate leaving frame

Every confirmed violation stores evidence from:

- Camera snapshot
- Screen snapshot
- CSV log entry (`evidence/violations.csv`)

## Project Structure

```
ExamProctorV2/
  proctor_app/
    core/
    detectors/
    io/
    ui/
    main.py
  evidence/
  requirements.txt
  run_overlay.bat
  run_fullscreen.bat
```

## Install

```bash
pip install -r requirements.txt
```

## Run

Overlay mode (small top-left style preview):

```bash
python -m proctor_app.main --view-mode overlay
```

Fullscreen mode:

```bash
python -m proctor_app.main --view-mode fullscreen
```

Batch shortcuts:

```bash
run_overlay.bat
run_fullscreen.bat
```

## Optional Arguments

- `--camera-index 0`
- `--monitor-index 1`
- `--overlay-width 420 --overlay-height 280`
- `--disable-device-detector`
- `--device-label "cell phone" --device-label "laptop"`

## Notes

- For robust tab/app switch detection, use `overlay` mode; it avoids full-screen self-capture noise.
- Phone detection uses YOLO (`ultralytics`) and may download model weights on first run.
- Evidence files are written to `ExamProctorV2/evidence/`.

## Push/Handover Checklist

- Keep these files committed:
  - `face_landmarker.task`
  - `darshanums.in/test_exam.html`
- Do not commit generated files:
  - `evidence/`
  - `yolov8n.pt` (auto-downloaded on first run if missing)
- On a new machine:
  1. Install Python.
  2. Run `pip install -r requirements.txt`.
  3. Run `run_overlay.bat` (or `run_fullscreen.bat`).
