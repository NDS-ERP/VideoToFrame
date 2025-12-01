import os
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import shutil
import zipfile
import uvicorn

app = FastAPI()

UPLOAD_DIR = "uploaded_videos"
FRAME_DIR = "frames"
ZIP_DIR = "zips"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)

# ---------------- 흐린 프레임 판단 ----------------
def is_blurry(frame, threshold=100.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

# ---------------- 중복 프레임 판단 ----------------
def is_similar(img1, img2, threshold=0.95):
    diff = cv2.absdiff(img1, img2)
    non_zero = np.count_nonzero(diff)
    total = diff.size
    similarity = 1 - (non_zero / total)
    return similarity > threshold

# ---------------- 프레임 추출 ----------------
def extract_frames(video_path, fps=5):
    video_id = str(uuid.uuid4())
    output_dir = os.path.join(FRAME_DIR, video_id)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(orig_fps // fps))

    prev_frame = None
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval != 0:
            frame_count += 1
            continue

        if is_blurry(frame):
            frame_count += 1
            continue

        if prev_frame is not None:
            if is_similar(frame, prev_frame, threshold=0.95):
                frame_count += 1
                continue

        save_path = os.path.join(output_dir, f"{saved_count:05d}.jpg")
        cv2.imwrite(save_path, frame)
        saved_count += 1
        prev_frame = frame
        frame_count += 1

    cap.release()
    return video_id, output_dir, saved_count

# ---------------- ZIP 생성 ----------------
def make_zip(folder_path, video_id):
    zip_path = os.path.join(ZIP_DIR, f"{video_id}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)
    return zip_path

# ---------------- 업로드 + 처리 API ----------------
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.{file.filename.split('.')[-1]}")
    with open(save_path, "wb") as f:
        f.write(await file.read())

    video_id, frame_dir, frame_count = extract_frames(save_path)

    zip_path = make_zip(frame_dir, video_id)

    # 업로드, 프레임 폴더는 정리
    shutil.rmtree(frame_dir)
    os.remove(save_path)

    return {"status": "success", "zip_url": f"/download/{video_id}", "frames_count": frame_count}

# ---------------- ZIP 다운로드 ----------------
@app.get("/download/{video_id}")
def download_zip(video_id: str):
    zip_path = os.path.join(ZIP_DIR, f"{video_id}.zip")
    if os.path.exists(zip_path):
        return FileResponse(zip_path, filename=f"{video_id}.zip", media_type='application/zip')
    return {"error": "ZIP file not found"}

# ---------------- 업로드 폼 ----------------
@app.get("/")
def form():
    return HTMLResponse("""
    <html>
        <body>
            <h2>영상 업로드 → 프레임 추출 → 중복 제거 → ZIP 다운로드</h2>
            <form action="/upload" enctype="multipart/form-data" method="post">
                <input type="file" name="file" accept="video/*">
                <button type="submit">프레임 추출</button>
            </form>
        </body>
    </html>
    """)

# ---------------- 서버 실행 ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
