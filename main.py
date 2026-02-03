import cv2
import numpy as np
import base64
import pytesseract
import io
import platform
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from docx import Document

# --- CONFIGURATION ---
# We check if we are on Windows. If yes, use local paths. 
# If on Linux (Render), we skip this and let it find the tools automatically.
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    POPPLER_PATH = r'C:\Program Files\poppler-25.12.0\Library\bin'
else:
    POPPLER_PATH = None 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImagePayload(BaseModel):
    image: str
    filter_type: str = "scan" 

# --- CV LOGIC ---

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    maxWidth = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    maxHeight = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def process_image_logic(image, filter_type="scan"):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image_small = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    kernel = np.ones((5,5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is not None and filter_type != "photo":
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    else:
        warped = orig

    if filter_type == "bw":
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    elif filter_type == "scan":
        if len(warped.shape) == 3: warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)
    elif filter_type == "enhance":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        warped = cv2.filter2D(warped, -1, kernel)
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        warped[:, :, 1] = cv2.multiply(warped[:, :, 1], 1.2)
        warped = cv2.cvtColor(warped, cv2.COLOR_HSV2BGR)

    return warped

@app.post("/process")
async def process_document(payload: ImagePayload):
    try:
        if "," in payload.image:
            header, encoded = payload.image.split(",", 1)
        else:
            header, encoded = "data:image/jpeg;base64", payload.image
            
        file_bytes = base64.b64decode(encoded)
        
        # Determine Source Type
        if "application/pdf" in header:
            # FIX: Only pass poppler_path if on Windows (not None)
            if POPPLER_PATH:
                 images = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)
            else:
                 images = convert_from_bytes(file_bytes) # Linux uses default path
            
            if len(images) > 0:
                img = np.array(images[0])[:, :, ::-1].copy()
        elif "officedocument" in header:
            doc = Document(io.BytesIO(file_bytes))
            text = "\n".join([p.text for p in doc.paragraphs])
            blank = np.zeros((800, 600, 3), np.uint8) + 255
            cv2.putText(blank, "DOCX Text Extracted", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            _, buf = cv2.imencode('.jpg', blank)
            return {"status": "success", "scanned_image": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}", "text": text}
        elif "text/plain" in header:
            text = file_bytes.decode("utf-8")
            blank = np.zeros((800, 600, 3), np.uint8) + 255
            _, buf = cv2.imencode('.jpg', blank)
            return {"status": "success", "scanned_image": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}", "text": text}
        else:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        processed = process_image_logic(img, payload.filter_type)
        
        gray_for_ocr = processed if len(processed.shape) == 2 else cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_for_ocr)

        _, buffer = cv2.imencode('.jpg', processed)
        processed_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

        return {
            "status": "success",
            "scanned_image": processed_base64,
            "text": text
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)