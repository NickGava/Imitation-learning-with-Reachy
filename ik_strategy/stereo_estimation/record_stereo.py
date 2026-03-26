'''
record_stereo.py
=============================================================================
Records a synchronized stereo video pair from Reachy's left and right cameras.

Output (same folder structure as existing pipeline):
  data/raw_data/subject_XXX/exercise_XXX/video_XXX_L.mp4   ← left camera
  data/raw_data/subject_XXX/exercise_XXX/video_XXX_R.mp4   ← right camera

Controls:
  SPACE - start / stop recording
  Q     - quit without saving (if not already recording)
          or stop recording and save (if recording)

Usage:
  python record_stereo.py
  >>> Subject number:  1
  >>> Exercise number: 2
  >>> Video number:    1
'''

import time
import threading
import cv2
from pathlib import Path
from reachy_sdk import ReachySDK

from config import DATA_ROOT

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ROBOT_IP     = "localhost" 
FOURCC       = cv2.VideoWriter_fourcc(*'mp4v')
DISPLAY_WIDTH = 640              # preview window width


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        subject_num  = int(input("Subject number:  ").strip())
        exercise_num = int(input("Exercise number: ").strip())
        video_num    = int(input("Video number:    ").strip())
    except ValueError:
        print("Error: all values must be integers.")
        return

    subject_name  = f"subject_{subject_num:03d}"
    exercise_name = f"exercise_{exercise_num:03d}"
    video_name    = f"video_{video_num:03d}"

    out_dir = DATA_ROOT / "raw_data" / subject_name / exercise_name
    out_dir.mkdir(parents=True, exist_ok=True)

    path_L = out_dir / f"{video_name}_L.mp4"
    path_R = out_dir / f"{video_name}_R.mp4"

    if path_L.exists() or path_R.exists():
        ans = input(f"File already exists: {video_name}_L/R.mp4 - overwrite? [y/N] ").strip().lower()
        if ans != 'y':
            print("Aborted.")
            return

    # Connect to Reachy
    print(f"Connecting to Reachy at {ROBOT_IP}…")
    reachy = ReachySDK(host=ROBOT_IP)
    print("Connected.\n")

    # Detect image size from first frame
    frame_L = None
    for _ in range(30):                 # wait up to ~1 s for a valid frame
        frame_L = reachy.left_camera.last_frame
        if frame_L is not None:
            break
        time.sleep(0.033)

    if frame_L is None:
        print("Error: could not read a frame from the left camera.")
        return

    h, w = frame_L.shape[:2]
    print(f"Camera resolution: {w}×{h}")
    print("Press SPACE to start recording, Q to quit.\n")

    writer_L = None
    writer_R = None
    recording = False
    frame_times = []

    try:
        while True:
            left  = reachy.left_camera.last_frame
            right = reachy.right_camera.last_frame

            if left is None or right is None:
                time.sleep(0.01)
                continue

            now = time.time()

            if recording:
                writer_L.write(left)
                writer_R.write(right)
                frame_times.append(now)

            # Preview (left camera)
            preview = left.copy()
            label   = "● REC" if recording else "SPACE=record  Q=quit"
            color   = (0, 0, 220) if recording else (0, 200, 0)
            cv2.putText(preview, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if recording:
                elapsed = now - frame_times[0]
                cv2.putText(preview, f"{elapsed:.1f}s  {len(frame_times)} frames",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            dh    = int(h * DISPLAY_WIDTH / w)
            shown = cv2.resize(preview, (DISPLAY_WIDTH, dh))
            cv2.imshow("Reachy - Stereo Recording (left camera)", shown)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if not recording:
                    # Start recording
                    # Use a placeholder FPS; we will re-compute and warn if needed.
                    fps_guess = 30.0
                    writer_L = cv2.VideoWriter(str(path_L), FOURCC, fps_guess, (w, h))
                    writer_R = cv2.VideoWriter(str(path_R), FOURCC, fps_guess, (w, h))
                    frame_times = []
                    recording   = True
                    print("Recording started…")
                else:
                    # Stop recording
                    recording = False
                    writer_L.release()
                    writer_R.release()
                    _report(frame_times, path_L, path_R)
                    break

            elif key == ord('q'):
                if recording:
                    recording = False
                    writer_L.release()
                    writer_R.release()
                    _report(frame_times, path_L, path_R)
                print("Quit.")
                break

    finally:
        cv2.destroyAllWindows()


def _report(frame_times, path_L, path_R):
    n = len(frame_times)
    if n > 1:
        duration = frame_times[-1] - frame_times[0]
        fps_real  = (n - 1) / duration if duration > 0 else 0.0
        print(f"\nRecording saved:")
        print(f"  {path_L}")
        print(f"  {path_R}")
        print(f"  Frames   : {n}")
        print(f"  Duration : {duration:.2f} s")
        print(f"  Avg FPS  : {fps_real:.1f}")
        if fps_real < 25:
            print("  ⚠  FPS lower than expected - check USB bandwidth or CPU load.")
    else:
        print("No frames recorded.")


if __name__ == "__main__":
    main()
