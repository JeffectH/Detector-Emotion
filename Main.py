from feat import Detector
import cv2
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from datetime import datetime
import os

print("=" * 60)
print("REAL-TIME VIDEO EMOTION DETECTOR WITH EXCEL EXPORT")
print("=" * 60)

# Initialize the detector
print("Loading emotion detection models...")
try:
    detector = Detector()
    print("✓ Detector loaded successfully!")
except Exception as e:
    print(f"Error loading detector: {e}")
    exit(1)

# Emotion labels dictionary
emotion_labels = {
    'anger': 'Anger',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'happiness': 'Happiness',
    'sadness': 'Sadness',
    'surprise': 'Surprise',
    'neutral': 'Neutral'
}

# Video source configuration (0 for webcam, or path to video file)
VIDEO_SOURCE = 0  # Change to video file path if needed, e.g., "path/to/video.mp4"

# Processing configuration - DEFINE BEFORE USE
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame for better performance
FIXED_TIME_STEP = 0.1  # Fixed time step in seconds for data recording (100ms)
DISPLAY_FPS = True
DISPLAY_ALL_EMOTIONS = True

# Data collection for Excel export
time_stamps = []
emotion_data = {emotion: [] for emotion in emotion_labels.keys()}

# Additional metrics data collection
gaze_data = {'gaze_x': [], 'gaze_y': []}  # Gaze tracking coordinates
head_pose_data = {'pitch': [], 'yaw': [], 'roll': []}  # Head pose angles
eye_data = {'blink_rate': [], 'eye_openness_left': [], 'eye_openness_right': []}  # Eye metrics

# Blink detection variables
blink_counter = 0
last_blink_check = 0
blink_threshold = 0.2  # Threshold for detecting closed eyes
frames_for_blink_rate = 30  # Calculate blink rate over 30 frames

# Performance counters
frame_count = 0
detection_count = 0
error_count = 0
skipped_frames = 0
start_time = time.time()
last_recorded_time = 0  # Track last recorded timestamp for fixed intervals


# Font setup for better text display
def setup_fonts():
    """Setup fonts with fallback options"""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font_large = ImageFont.truetype(font_path, 28)
                font_medium = ImageFont.truetype(font_path, 20)
                font_small = ImageFont.truetype(font_path, 14)
                print(f"✓ Loaded font: {font_path}")
                return font_large, font_medium, font_small
            except:
                continue

    print("⚠ Using default font")
    default = ImageFont.load_default()
    return default, default, default


font_large, font_medium, font_small = setup_fonts()


def save_to_excel(filename=None):
    """Save all collected emotion data to Excel with comprehensive charts"""

    if len(time_stamps) == 0:
        print("\n⚠ No data to save! Process some frames first.")
        return False

    # Define save directory (change this path as needed)
    SAVE_DIRECTORY = os.path.join(os.path.expanduser("~"), "Data analysis")  # Save to Data analysis
    # Alternative options:
    # SAVE_DIRECTORY = "C:/EmotionData"  # Specific folder
    # SAVE_DIRECTORY = os.getcwd()  # Current working directory

    # Create directory if it doesn't exist
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)

    if filename is None:
        filename = f"emotion_video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    # Full path for saving
    full_path = os.path.join(SAVE_DIRECTORY, filename)

    print(f"\n📊 Saving data to Excel...")
    print(f"📁 Location: {full_path}")

    # Create DataFrame with all metrics - with Russian translations
    df_data = {'Time (seconds) [Время (сек)]': time_stamps}

    # Add emotion data with Russian translations
    emotion_translations = {
        'Anger': 'Гнев',
        'Disgust': 'Отвращение',
        'Fear': 'Страх',
        'Happiness': 'Радость',
        'Sadness': 'Грусть',
        'Surprise': 'Удивление',
        'Neutral': 'Нейтральное'
    }

    for emotion in emotion_labels.keys():
        if len(emotion_data[emotion]) == len(time_stamps):
            emotion_eng = emotion_labels[emotion]
            emotion_rus = emotion_translations.get(emotion_eng, emotion_eng)
            column_name = f"{emotion_eng} [{emotion_rus}] (%)"
            df_data[column_name] = [x * 100 for x in emotion_data[emotion]]

    # Add gaze tracking data with Russian
    if len(gaze_data['gaze_x']) == len(time_stamps):
        df_data['Gaze_X [Взгляд_X]'] = gaze_data['gaze_x']
        df_data['Gaze_Y [Взгляд_Y]'] = gaze_data['gaze_y']

    # Add head pose data with Russian
    if len(head_pose_data['pitch']) == len(time_stamps):
        df_data['Head_Pitch [Наклон головы]'] = head_pose_data['pitch']
        df_data['Head_Yaw [Поворот головы]'] = head_pose_data['yaw']
        df_data['Head_Roll [Крен головы]'] = head_pose_data['roll']

    # Add eye metrics with Russian
    if len(eye_data['blink_rate']) == len(time_stamps):
        df_data['Blink_Rate [Частота морганий]'] = eye_data['blink_rate']
        df_data['Eye_Open_Left [Левый глаз] (%)'] = [x * 100 for x in eye_data['eye_openness_left']]
        df_data['Eye_Open_Right [Правый глаз] (%)'] = [x * 100 for x in eye_data['eye_openness_right']]

    df = pd.DataFrame(df_data)

    # Calculate statistics with Russian headers
    stats_data = {
        'Emotion [Эмоция]': [],
        'Mean [Среднее] (%)': [],
        'Max [Максимум] (%)': [],
        'Min [Минимум] (%)': [],
        'Std Dev [Откл.] (%)': []
    }

    for emotion in emotion_labels.keys():
        emotion_eng = emotion_labels[emotion]
        emotion_rus = emotion_translations.get(emotion_eng, emotion_eng)

        # Find the column with this emotion
        for col in df.columns:
            if emotion_eng in col and '[' in col:  # Find the emotion column with translation
                stats_data['Emotion [Эмоция]'].append(f"{emotion_eng} [{emotion_rus}]")
                stats_data['Mean [Среднее] (%)'].append(df[col].mean())
                stats_data['Max [Максимум] (%)'].append(df[col].max())
                stats_data['Min [Минимум] (%)'].append(df[col].min())
                stats_data['Std Dev [Откл.] (%)'].append(df[col].std())
                break

    df_stats = pd.DataFrame(stats_data)

    # Save to Excel with multiple sheets
    with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
        # Raw data sheet
        df.to_excel(writer, sheet_name='Emotion Data', index=False)

        # Statistics sheet - add descriptions
        df_stats.to_excel(writer, sheet_name='Statistics', index=False)

        # Get workbook and data worksheet
        workbook = writer.book
        data_sheet = writer.sheets['Emotion Data']
        stats_sheet = writer.sheets['Statistics']

        # Add metric descriptions in statistics sheet
        descriptions_row = len(df_stats) + 3
        descriptions = [
            ['', ''],
            ['Описание метрик:', ''],
            ['Mean [Среднее]', 'Среднее значение эмоции за весь период записи'],
            ['Max [Максимум]', 'Максимальное проявление эмоции (пиковое значение)'],
            ['Min [Минимум]', 'Минимальное значение эмоции'],
            ['Std Dev [Откл.]', 'Стандартное отклонение - показывает изменчивость/стабильность эмоции'],
            ['', ''],
            ['Интерпретация:', ''],
            ['Высокое среднее (>30%)', 'Эмоция часто проявлялась'],
            ['Высокое отклонение (>20%)', 'Эмоция нестабильна, есть всплески'],
            ['Низкое отклонение (<10%)', 'Эмоция стабильна на протяжении времени'],
        ]

        for i, (label, description) in enumerate(descriptions):
            stats_sheet.cell(row=descriptions_row + i, column=1, value=label)
            stats_sheet.cell(row=descriptions_row + i, column=2, value=description)

        # Format columns in data sheet
        for column in data_sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 20)
            data_sheet.column_dimensions[column_letter].width = adjusted_width

        # Create comprehensive chart for all emotions
        main_chart = LineChart()
        main_chart.title = "All Emotions Over Time"
        main_chart.style = 10
        main_chart.y_axis.title = 'Probability (%)'
        main_chart.x_axis.title = 'Time (seconds)'
        main_chart.height = 15
        main_chart.width = 30

        # Add all emotion data to main chart
        data = Reference(data_sheet, min_col=2, min_row=1,
                         max_col=len(emotion_labels) + 1, max_row=len(df) + 1)
        categories = Reference(data_sheet, min_col=1, min_row=2, max_row=len(df) + 1)
        main_chart.add_data(data, titles_from_data=True)
        main_chart.set_categories(categories)

        # Position main chart - moved further right to avoid data columns
        data_sheet.add_chart(main_chart, "M2")

        # Create individual charts for each emotion with proper spacing
        chart_row = 35  # Start lower to avoid overlap with main chart
        chart_positions = ['M', 'AC']  # Two columns, well-spaced

        for idx, (emotion_key, emotion_label) in enumerate(emotion_labels.items(), start=2):
            individual_chart = LineChart()
            individual_chart.title = f"{emotion_label} Timeline"
            individual_chart.style = 10
            individual_chart.y_axis.title = 'Probability (%)'
            individual_chart.x_axis.title = 'Time (seconds)'
            individual_chart.height = 12
            individual_chart.width = 18

            # Add data for this specific emotion
            data = Reference(data_sheet, min_col=idx, min_row=1,
                             max_col=idx, max_row=len(df) + 1)
            categories = Reference(data_sheet, min_col=1, min_row=2, max_row=len(df) + 1)
            individual_chart.add_data(data, titles_from_data=True)
            individual_chart.set_categories(categories)

            # Calculate position with proper spacing
            col_index = (idx - 2) % 2
            col_letter = chart_positions[col_index]

            data_sheet.add_chart(individual_chart, f"{col_letter}{chart_row}")

            if col_index == 1:  # Move to next row after second column
                chart_row += 25  # Increased spacing between rows

        # Create additional charts for new metrics with proper positioning

        # Calculate starting row for additional metrics charts
        additional_charts_row = chart_row + 30  # Add space after emotion charts

        # Gaze tracking chart
        if 'Gaze_X' in df.columns and 'Gaze_Y' in df.columns:
            gaze_chart = LineChart()
            gaze_chart.title = "Gaze Tracking Over Time"
            gaze_chart.style = 10
            gaze_chart.y_axis.title = 'Gaze Coordinates'
            gaze_chart.x_axis.title = 'Time (seconds)'
            gaze_chart.height = 12
            gaze_chart.width = 18

            # Find column indices for gaze data
            gaze_x_col = df.columns.get_loc('Gaze_X') + 1
            gaze_y_col = df.columns.get_loc('Gaze_Y') + 1

            data = Reference(data_sheet, min_col=gaze_x_col, min_row=1,
                             max_col=gaze_y_col, max_row=len(df) + 1)
            categories = Reference(data_sheet, min_col=1, min_row=2, max_row=len(df) + 1)
            gaze_chart.add_data(data, titles_from_data=True)
            gaze_chart.set_categories(categories)
            data_sheet.add_chart(gaze_chart, f"M{additional_charts_row}")

        # Head pose chart
        if 'Head_Pitch' in df.columns:
            head_chart = LineChart()
            head_chart.title = "Head Pose Over Time"
            head_chart.style = 10
            head_chart.y_axis.title = 'Angle (degrees)'
            head_chart.x_axis.title = 'Time (seconds)'
            head_chart.height = 12
            head_chart.width = 18

            pitch_col = df.columns.get_loc('Head_Pitch') + 1
            roll_col = df.columns.get_loc('Head_Roll') + 1

            data = Reference(data_sheet, min_col=pitch_col, min_row=1,
                             max_col=roll_col, max_row=len(df) + 1)
            categories = Reference(data_sheet, min_col=1, min_row=2, max_row=len(df) + 1)
            head_chart.add_data(data, titles_from_data=True)
            head_chart.set_categories(categories)
            data_sheet.add_chart(head_chart, f"AC{additional_charts_row}")

        # Blink rate chart
        if 'Blink_Rate' in df.columns:
            blink_chart = LineChart()
            blink_chart.title = "Blink Rate and Eye Openness"
            blink_chart.style = 10
            blink_chart.y_axis.title = 'Blinks/min | Openness (%)'
            blink_chart.x_axis.title = 'Time (seconds)'
            blink_chart.height = 12
            blink_chart.width = 18

            blink_col = df.columns.get_loc('Blink_Rate') + 1

            data = Reference(data_sheet, min_col=blink_col, min_row=1,
                             max_col=blink_col, max_row=len(df) + 1)
            categories = Reference(data_sheet, min_col=1, min_row=2, max_row=len(df) + 1)
            blink_chart.add_data(data, titles_from_data=True)
            blink_chart.set_categories(categories)
            data_sheet.add_chart(blink_chart, f"M{additional_charts_row + 25}")

        # Create summary chart in statistics sheet
        summary_chart = LineChart()
        summary_chart.title = "Emotion Statistics Comparison"
        summary_chart.style = 10
        stats_sheet.add_chart(summary_chart, "G2")

    print(f"✅ Excel file saved: {filename}")
    print(f"📍 Full path: {full_path}")
    print(f"   📈 {len(df)} data points")
    print(f"   📊 {len(emotion_labels) + 1} charts created")
    print(f"   📋 2 sheets (Data + Statistics)")

    # Open file location in explorer (Windows)
    if os.name == 'nt':  # Windows
        os.startfile(SAVE_DIRECTORY)

    return True


def process_video_frame(frame, current_time):
    """Process a single video frame and collect emotion data"""
    global detection_count, error_count, last_recorded_time, blink_counter, last_blink_check

    display_frame = frame.copy()
    should_record = False

    # Check if enough time has passed for next data point (fixed time step)
    if current_time - last_recorded_time >= FIXED_TIME_STEP:
        should_record = True

    try:
        # Save frame temporarily for detection
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        # Detect emotions and additional features
        results = detector.detect_image(temp_path)

        # Clean up temp file immediately
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if results is not None and not results.empty and len(results) > 0:
            detection_count += 1
            first_result = results.iloc[0]

            # Get face coordinates
            x, y, w, h = 50, 50, 200, 200  # Default values

            if 'FaceRectX' in results.columns:
                x = int(first_result.get('FaceRectX', x))
                y = int(first_result.get('FaceRectY', y))
                w = int(first_result.get('FaceRectWidth', w))
                h = int(first_result.get('FaceRectHeight', h))

            # Draw face rectangle
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Extract additional metrics
            current_gaze = {'x': 0, 'y': 0}
            current_head_pose = {'pitch': 0, 'yaw': 0, 'roll': 0}
            current_eye_openness = {'left': 1.0, 'right': 1.0}

            # Gaze tracking (if available in results)
            if 'gaze_x' in results.columns and 'gaze_y' in results.columns:
                current_gaze['x'] = float(first_result.get('gaze_x', 0))
                current_gaze['y'] = float(first_result.get('gaze_y', 0))

                # Draw gaze direction arrow
                gaze_end_x = int(x + w // 2 + current_gaze['x'] * 100)
                gaze_end_y = int(y + h // 2 + current_gaze['y'] * 100)
                cv2.arrowedLine(display_frame, (x + w // 2, y + h // 2),
                                (gaze_end_x, gaze_end_y), (255, 0, 255), 2)

            # Head pose estimation (pitch, yaw, roll)
            if 'pitch' in results.columns:
                current_head_pose['pitch'] = float(first_result.get('pitch', 0))
                current_head_pose['yaw'] = float(first_result.get('yaw', 0))
                current_head_pose['roll'] = float(first_result.get('roll', 0))

            # Eye openness for blink detection
            if 'AU43' in results.columns:  # AU43 = Eye closure
                eye_closure = float(first_result.get('AU43', 0))
                current_eye_openness['left'] = 1.0 - eye_closure
                current_eye_openness['right'] = 1.0 - eye_closure

                # Detect blinks
                if eye_closure > blink_threshold and current_time - last_blink_check > 0.1:
                    blink_counter += 1
                    last_blink_check = current_time

            # Calculate blink rate (blinks per minute)
            time_window = min(current_time, 60)  # Use up to 60 seconds
            current_blink_rate = (blink_counter / max(time_window, 1)) * 60

            # Collect emotion data only at fixed time intervals
            current_emotions = {}
            for emotion in emotion_labels.keys():
                if emotion in results.columns:
                    score = float(first_result[emotion])
                    current_emotions[emotion] = score
                    if should_record:
                        emotion_data[emotion].append(score)
                else:
                    if should_record:
                        emotion_data[emotion].append(0.0)

            # Record all metrics at fixed intervals
            if should_record:
                time_stamps.append(round(current_time, 1))
                last_recorded_time = current_time

                # Record additional metrics
                gaze_data['gaze_x'].append(current_gaze['x'])
                gaze_data['gaze_y'].append(current_gaze['y'])
                head_pose_data['pitch'].append(current_head_pose['pitch'])
                head_pose_data['yaw'].append(current_head_pose['yaw'])
                head_pose_data['roll'].append(current_head_pose['roll'])
                eye_data['blink_rate'].append(current_blink_rate)
                eye_data['eye_openness_left'].append(current_eye_openness['left'])
                eye_data['eye_openness_right'].append(current_eye_openness['right'])

            # Find dominant emotion
            if current_emotions:
                dominant_emotion = max(current_emotions, key=current_emotions.get)
                dominant_score = current_emotions[dominant_emotion]

                # Convert to PIL for better text rendering
                pil_img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)

                # Get frame dimensions
                frame_height, frame_width = display_frame.shape[:2]

                # Draw dominant emotion above face box
                emotion_text = f"{emotion_labels[dominant_emotion]}: {dominant_score:.1%}"
                draw.text((x, y - 35), emotion_text, font=font_large, fill=(0, 255, 0))

                # Draw additional metrics on the right side
                metrics_x = frame_width - 250
                metrics_y = 60

                draw.text((metrics_x, metrics_y), "Additional Metrics:", font=font_medium, fill=(255, 255, 255))
                metrics_y += 25

                # Gaze direction
                gaze_text = f"Gaze: ({current_gaze['x']:.2f}, {current_gaze['y']:.2f})"
                draw.text((metrics_x, metrics_y), gaze_text, font=font_small, fill=(255, 0, 255))
                metrics_y += 18

                # Head pose
                pose_text = f"Head: P:{current_head_pose['pitch']:.1f}° Y:{current_head_pose['yaw']:.1f}° R:{current_head_pose['roll']:.1f}°"
                draw.text((metrics_x, metrics_y), pose_text, font=font_small, fill=(0, 255, 255))
                metrics_y += 18

                # Blink rate
                blink_text = f"Blinks/min: {current_blink_rate:.1f}"
                draw.text((metrics_x, metrics_y), blink_text, font=font_small, fill=(255, 255, 0))
                metrics_y += 18

                # Eye openness
                eye_text = f"Eyes: L:{current_eye_openness['left']:.1%} R:{current_eye_openness['right']:.1%}"
                draw.text((metrics_x, metrics_y), eye_text, font=font_small, fill=(0, 255, 0))

                # Draw all emotions in sidebar if enabled
                if DISPLAY_ALL_EMOTIONS:
                    y_offset = 60
                    draw.text((10, y_offset - 20), "Emotions:", font=font_medium, fill=(255, 255, 255))

                    for emotion, score in current_emotions.items():
                        score_text = f"{emotion_labels[emotion]}: {score:.1%}"

                        # Color code based on intensity
                        if score > 0.5:
                            color = (0, 255, 0)  # Green for high
                        elif score > 0.3:
                            color = (255, 255, 0)  # Yellow for medium
                        else:
                            color = (200, 200, 200)  # Gray for low

                        if emotion == dominant_emotion:
                            color = (0, 255, 100)  # Highlight dominant

                        draw.text((15, y_offset), score_text, font=font_small, fill=color)
                        y_offset += 18

                # Convert back to OpenCV format
                display_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            return display_frame, True
        else:
            # No face detected
            cv2.putText(display_frame, "No face detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return display_frame, False

    except Exception as e:
        error_count += 1
        cv2.putText(display_frame, f"Error: {str(e)[:30]}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return display_frame, False


# Main video processing loop
def main():
    global frame_count, skipped_frames, detection_count, error_count, start_time, last_recorded_time
    global blink_counter, last_blink_check

    # Open video source
    print(f"\n🎥 Opening video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("❌ Error: Could not open video source!")
        print("   Try: 0 for webcam, or provide a valid video file path")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default FPS if cannot be determined

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✅ Video opened successfully!")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.1f}")
    print(f"\n📌 Controls")
    print("   'S' - Save data to Excel")
    print("   'R' - Reset data collection")
    print("   'Q' - Quit")
    print("\n🔄 Processing")

    # FPS calculation variables
    fps_start_time = time.time()
    fps_frame_count = 0
    display_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n⚠ End of video or camera disconnected")
            break

        frame_count += 1
        current_time = time.time() - start_time

        # Calculate display FPS
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            display_fps = fps_frame_count
            fps_frame_count = 0
            fps_start_time = time.time()

        # Process frame based on sampling rate
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            display_frame, detected = process_video_frame(frame, current_time)
        else:
            display_frame = frame
            skipped_frames += 1

        # Get frame dimensions for UI elements
        frame_height, frame_width = display_frame.shape[:2]

        # Add UI overlay with full width background
        overlay = display_frame.copy()

        # Semi-transparent background for stats - full width
        cv2.rectangle(overlay, (0, 5), (frame_width, 40), (50, 50, 50), -1)
        display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)

        # Display statistics
        stats_text = f"FPS: {display_fps} | Frames: {frame_count} | Detections: {detection_count}"
        cv2.putText(display_frame, stats_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Control buttons overlay with bilingual text
        button_y = frame_height - 60
        cv2.rectangle(display_frame, (10, button_y), (420, frame_height - 10), (80, 80, 80), -1)
        cv2.putText(display_frame, "[S] Save/ | [R] Reset | [Q] Quit",
                    (20, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Data collection indicator
        if len(time_stamps) > 0:
            data_text = f"Data points: {len(time_stamps)}"
            cv2.putText(display_frame, data_text, (frame_width - 150, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the frame
        cv2.imshow('Real-time Emotion Detection', display_frame)

        # Handle keyboard input - support both English and Russian layouts
        key = cv2.waitKey(1) & 0xFF

        # Support both English and Russian keyboard layouts
        if key == ord('q') or key == ord('Q') or key == ord('й') or key == ord('Й'):
            print("\n👋 Quitting...")
            break

        elif key == ord('s') or key == ord('S') or key == ord('ы') or key == ord('Ы'):
            if save_to_excel():
                # Show confirmation on screen
                cv2.putText(display_frame, "DATA SAVED!",
                            (frame_width // 2 - 150, frame_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.imshow('Real-time Emotion Detection', display_frame)
                cv2.waitKey(1000)

        elif key == ord('r') or key == ord('R') or key == ord('к') or key == ord('К'):
            # Reset all data collections
            time_stamps.clear()
            for emotion in emotion_data:
                emotion_data[emotion].clear()

            # Reset additional metrics
            gaze_data['gaze_x'].clear()
            gaze_data['gaze_y'].clear()
            head_pose_data['pitch'].clear()
            head_pose_data['yaw'].clear()
            head_pose_data['roll'].clear()
            eye_data['blink_rate'].clear()
            eye_data['eye_openness_left'].clear()
            eye_data['eye_openness_right'].clear()

            # Reset counters
            detection_count = 0
            error_count = 0
            frame_count = 0
            skipped_frames = 0
            blink_counter = 0
            start_time = time.time()
            last_recorded_time = 0
            last_blink_check = 0
            print("\n🔄 All data collection reset!")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Auto-save if data exists
    if len(time_stamps) > 0:
        print("\n💾 Auto-saving collected data...")
        save_to_excel()

    # Final statistics
    print("\n" + "=" * 60)
    print("📊 FINAL STATISTICS:")
    print(f"   Total frames: {frame_count}")
    print(f"   Processed frames: {frame_count - skipped_frames}")
    print(f"   Skipped frames: {skipped_frames}")
    print(f"   Successful detections: {detection_count}")
    print(f"   Errors: {error_count}")
    if frame_count > 0:
        detection_rate = detection_count / max(1, (frame_count - skipped_frames)) * 100
        print(f"   Detection rate: {detection_rate:.1f}%")
    print(f"   Data points collected: {len(time_stamps)}")
    print(f"   Total runtime: {time.time() - start_time:.1f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()