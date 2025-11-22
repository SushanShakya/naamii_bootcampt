from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import shutil
import uuid

from fastapi.responses import StreamingResponse

from src.agri import run_agri_model

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


async def handle_file_upload(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    ext = file.filename.split(".")[1]

    filename = uuid.uuid4()

    save_path = os.path.join(UPLOAD_DIR, f"{filename}.{ext}")

    # Save using streamed copy (efficient for large files)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename, "saved_to": save_path}


@app.post("/sam")
async def run_sam(file: UploadFile = File(...)):
    details = await handle_file_upload(file)
    return details


@app.post("/agri")
async def run_agri(pre: UploadFile = File(...), post: UploadFile = File(...)):
    pre_file = await handle_file_upload(pre)
    post_file = await handle_file_upload(post)

    return StreamingResponse(
        run_agri_model(pre_file.get("saved_to"), post_file.get("saved_to")),
        media_type="application/json",
    )


@app.post("/infra")
async def run_infra(file: UploadFile = File(...)):
    f = await handle_file_upload(file)
    path = f.get("saved_to")
    return f
