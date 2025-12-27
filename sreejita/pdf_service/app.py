# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import asyncio
import os

from renderer import render_pdf_from_url

app = FastAPI(
    title="Sreejita PDF Service",
    version="1.0",
)


class PDFRequest(BaseModel):
    html_url: HttpUrl


@app.post("/render-pdf")
async def render_pdf(req: PDFRequest):
    try:
        pdf_path = await render_pdf_from_url(req.html_url)

        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename="report.pdf",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {str(e)}",
        )
