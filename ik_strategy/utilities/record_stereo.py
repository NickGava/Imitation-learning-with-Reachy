'''
record_stereo.py
=============================================================================
Records a synchronized stereo video pair from Reachy's left and right cameras.

Output (same folder structure as existing pipeline):
  data/raw_data/subject_XXX/exercise_XXX/video_XXX_L.mp4   <- left camera
  data/raw_data/subject_XXX/exercise_XXX/video_XXX_R.mp4   <- right camera

Controls:
  SPACE - start / stop recording
  Q     - quit without saving (if not already recording)
          or stop recording and save (if recording)

Usage:
  python record_stereo.py                                    (prompt interattivo)
  python record_stereo.py --subject 1 --exercise 2 --video 1 (parte subito)
  python record_stereo.py --autostart                        (prompt + parte subito)
'''

import argparse
import time
import cv2
from reachy_sdk import ReachySDK

from utilities.config import DATA_ROOT
from utilities.ask_inputs import ask_inputs

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ROBOT_IP     = "10.59.1.20" 
FOURCC       = cv2.VideoWriter_fourcc(*'mp4v')
DISPLAY_WIDTH = 500              # preview window width

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Record synchronized stereo video from Reachy.')
    parser.add_argument('--subject',  type=int, default=None, help='Subject number (e.g. 1)')
    parser.add_argument('--exercise', type=int, default=None, help='Exercise number (e.g. 2)')
    parser.add_argument('--video', type=int, default=None, help='video number (e.g. 2)')
    parser.add_argument('--autostart', action='store_true', help='Start recording immediately without pressing SPACE')
    args = parser.parse_args()

    if args.subject is not None and args.exercise is not None and args.video is not None:
        subject_name  = f'subject_{args.subject:03d}'
        exercise_name = f'exercise_{args.exercise:03d}'
        video_name    = f'video_{args.video:03d}'
        autostart     = True   # se tutti gli argomenti sono passati, parte subito
    else:
        subject_name, exercise_name, video_name = ask_inputs()
        autostart = args.autostart

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
    print(f"Camera resolution: {w}x{h}")
    if autostart:
        print("Auto-start: recording begins immediately. Press SPACE or Q to stop.\n")
    else:
        print("Press SPACE to start recording, Q to quit.\n")

    writer_L = None
    writer_R = None
    recording = False
    frame_times = []

    middle_clicked = [False]  # lista per mutabilità nel closure

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MBUTTONDOWN:
            middle_clicked[0] = True

    cv2.namedWindow("Reachy - Stereo Recording (left camera)")
    cv2.setMouseCallback("Reachy - Stereo Recording (left camera)", on_mouse)

    # Auto-start: avvia la registrazione prima di entrare nel loop
    if autostart:
        fps_guess = 30.0
        writer_L  = cv2.VideoWriter(str(path_L), FOURCC, fps_guess, (w, h))
        writer_R  = cv2.VideoWriter(str(path_R), FOURCC, fps_guess, (w, h))
        frame_times = []
        recording   = True
        print("Recording started…")

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
            cv2.putText(preview, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if recording:
                elapsed = now - frame_times[0]
                cv2.putText(preview, f"{elapsed:.1f}s  {len(frame_times)} frames", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            dh    = int(h * DISPLAY_WIDTH / w)
            shown = cv2.resize(preview, (DISPLAY_WIDTH, dh))
            cv2.imshow("Reachy - Stereo Recording (left camera)", shown)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') or middle_clicked[0]:
                middle_clicked[0] = False
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