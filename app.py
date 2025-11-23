from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import shutil
import uuid

from fastapi.responses import StreamingResponse

from src.inference import run_infra_model
from src.agri2 import run_agri2
from src.agri import run_agri_model

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ allow requests from any domain
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)

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


@app.post("/agri")
async def run_agri(pre: UploadFile = File(...), post: UploadFile = File(...)):
    pre_file = await handle_file_upload(pre)
    post_file = await handle_file_upload(post)

    # VGG model
    return StreamingResponse(
        run_agri_model(pre_file.get("saved_to"), post_file.get("saved_to")),
        media_type="application/json",
    )


@app.post("/agri/b4")
async def run_agri(pre: UploadFile = File(...), post: UploadFile = File(...)):
    pre_file = await handle_file_upload(pre)
    post_file = await handle_file_upload(post)

    return StreamingResponse(
        run_agri2(pre_file.get("saved_to"), post_file.get("saved_to"), "b4"),
        media_type="application/json",
    )


@app.post("/agri/resnet")
async def run_agri(pre: UploadFile = File(...), post: UploadFile = File(...)):
    pre_file = await handle_file_upload(pre)
    post_file = await handle_file_upload(post)

    return StreamingResponse(
        run_agri2(pre_file.get("saved_to"), post_file.get("saved_to"), "resnet"),
        media_type="application/json",
    )


@app.post("/infra")
async def run_infra(file: UploadFile = File(...)):
    f = await handle_file_upload(file)
    path = f.get("saved_to")

    return StreamingResponse(
        run_infra_model(path),
        media_type="application/json",
    )
