import rppg
import time
import csv
from datetime import datetime
import cv2
import numpy as np

def calculate_bvp_snr(bvp_signal, fps=30.0):
    """Menghitung Signal-to-Noise Ratio (SNR dalam dB) pada rentang detak jantung (45-180 BPM)."""
    if bvp_signal is None or len(bvp_signal) < 30:
        return 0.0
    sig = np.asarray(bvp_signal, dtype=np.float32)
    sig = sig - np.mean(sig)
    
    # Hitung spektrum frekuensi menggunakan FFT
    fft_vals = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fps)
    
    # Rentang frekuensi denyut jantung manusia normal: 0.75 Hz (45 BPM) - 3.0 Hz (180 BPM)
    hr_mask = (freqs >= 0.75) & (freqs <= 3.0)
    noise_mask = (freqs > 0.2) & (~hr_mask)
    
    signal_power = np.sum(fft_vals[hr_mask])
    noise_power = np.sum(fft_vals[noise_mask]) + 1e-8
    
    snr_db = 10.0 * np.log10(signal_power / noise_power + 1e-8)
    return round(float(snr_db), 2)

def draw_live_bvp(bvp_signal, snr_db, motion_score, phase_label, width, height=140):
    """Membuat panel grafik gelombang BVP real-time + metrik SNR dan gerakan kepala."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    
    cv2.line(panel, (0, height // 2), (width, height // 2), (45, 45, 45), 1)
    header_text = f"LIVE BVP | SNR: {snr_db:.1f} dB | Motion: {motion_score:.1f} px | Phase: {phase_label}"
    cv2.putText(panel, header_text, (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1)

    if bvp_signal is not None and len(bvp_signal) >= 10:
        sig = np.asarray(bvp_signal, dtype=np.float32)
        sig_min, sig_max = np.min(sig), np.max(sig)
        if sig_max - sig_min > 1e-6:
            sig_norm = (sig - sig_min) / (sig_max - sig_min)
        else:
            sig_norm = np.full_like(sig, 0.5)

        margin_y = 32
        usable_h = height - (margin_y * 2)
        xs = np.linspace(10, width - 10, len(sig_norm)).astype(np.int32)
        ys = (height - margin_y - (sig_norm * usable_h)).astype(np.int32)

        pts = np.column_stack((xs, ys)).reshape((-1, 1, 2))
        cv2.polylines(panel, [pts], isClosed=False, color=(0, 255, 120), thickness=2, lineType=cv2.LINE_AA)

    return panel

model = rppg.Model()
file_csv = "hasil_eksperimen_rppg.csv"

with open(file_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Waktu", "Detik_Ke", "Fase", "BPM", "SNR_dB", "Motion_Px"])

    with model.video_capture(0):
        start_time = time.time()
        last_process_time = 0
        current_hr = None
        current_snr = 0.0
        latest_bvp = None
        
        prev_center = None
        motion_buffer = []
        current_phase = "Stationary"  # Default fase diam
        privacy_blur = True           # Auto-blur wajah aktif secara default

        print("Kamera aktif (Privacy Auto-Blur ON).")
        print("Kontrol Keyboard: [1] Fase Diam | [2] Fase Gerak | [b] Toggle Blur | [q] Selesai")

        for frame, box in model.preview:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            now = time.time()

            # Hitung pergeseran titik tengah wajah antar frame (Motion Artifact Score)
            if box is not None:
                y1, y2 = int(box[0][0]), int(box[0][1])
                x1, x2 = int(box[1][0]), int(box[1][1])
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                
                if prev_center is not None:
                    dist = np.hypot(cx - prev_center[0], cy - prev_center[1])
                    motion_buffer.append(dist)
                prev_center = (cx, cy)

            # Ambil potongan sinyal BVP 5 detik terakhir
            try:
                bvp_slice, _ = model.bvp(start=-5)
                if len(bvp_slice) > 0:
                    latest_bvp = bvp_slice
            except Exception:
                pass

            # Hitung BPM, SNR, dan rata-rata gerakan tiap 1 detik
            if now - last_process_time > 1.0:
                result = model.hr(start=-10)
                if result and result.get('hr') is not None:
                    current_hr = float(result['hr'])
                    current_snr = calculate_bvp_snr(latest_bvp, fps=30.0)
                    avg_motion = round(float(np.mean(motion_buffer)), 2) if motion_buffer else 0.0
                    motion_buffer.clear()

                    elapsed = round(now - start_time, 1)
                    timestamp_str = datetime.now().strftime("%H:%M:%S")

                    print(f"[{timestamp_str}] {elapsed}s | {current_phase} | HR: {current_hr:.1f} BPM | SNR: {current_snr} dB | Motion: {avg_motion} px")
                    writer.writerow([timestamp_str, elapsed, current_phase, round(current_hr, 1), current_snr, avg_motion])
                    f.flush()

                last_process_time = now

            # Terapkan Auto-Blur pada area wajah & gambar bounding box + info BPM
            if box is not None:
                h_frame, w_frame, _ = frame.shape
                y1, y2 = max(0, int(box[0][0])), min(h_frame, int(box[0][1]))
                x1, x2 = max(0, int(box[1][0])), min(w_frame, int(box[1][1]))

                if privacy_blur and y2 > y1 and x2 > x1:
                    face_roi = frame[y1:y2, x1:x2]
                    # Kernel (99, 99) membuat wajah buram total untuk menjaga privasi
                    blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                    frame[y1:y2, x1:x2] = blurred_face

                color_box = (0, 255, 0) if current_phase == "Stationary" else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)

                if current_hr is not None:
                    cv2.putText(frame, f"HR: {current_hr:.1f} BPM", (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_box, 2)

            h, w, _ = frame.shape
            display_motion = float(np.mean(motion_buffer)) if motion_buffer else 0.0
            bvp_panel = draw_live_bvp(latest_bvp, current_snr, display_motion, current_phase, width=w, height=140)
            combined_frame = np.vstack((frame, bvp_panel))

            cv2.imshow("rPPG Research Monitor - Privacy Mode", combined_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_phase = "Stationary"
            elif key == ord('2'):
                current_phase = "Motion"
            elif key == ord('b'):
                privacy_blur = not privacy_blur

cv2.destroyAllWindows()
print(f"\nSelesai. Data eksperimen tersimpan di '{file_csv}'.")