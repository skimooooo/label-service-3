import base64
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from label_gen import generate_label


app = FastAPI(
    title="TrackFlow MECHMAXX Label Image Generator",
    version="1.0.0",
)


class Recipient(BaseModel):
    name: str
    line1: str = ""
    line2: str = ""
    line3: str = ""
    line4: str = ""
    phone: Optional[str] = ""


class GenerateLabelRequest(BaseModel):
    recipient: Recipient
    tracking_number: str = "TF-QZ6HTDVPRH"
    response_format: str = "image"  # image, url, base64


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "TrackFlow MECHMAXX Label Image Generator",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/generate-label")
def generate(req: GenerateLabelRequest):
    try:
        img = generate_label(
            recipient=req.recipient.model_dump(),
            tracking_number=req.tracking_number,
        )

        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        fmt = (req.response_format or "image").lower().strip()

        if fmt == "base64":
            return JSONResponse({
                "image_base64": base64.b64encode(png_bytes).decode("utf-8"),
                "mime_type": "image/png",
            })

        # "image" and "url" both return a real PNG image.
        # Swagger displays this directly, and Lovable can fetch it as an image/blob.
        return StreamingResponse(BytesIO(png_bytes), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
