import asyncio
import os
from pathlib import Path

import f5_tts
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from f5_tts.api import F5TTS

from config import TTS_NFE_STEP

app = FastAPI()

# Load F5-TTS English Base model
print("Loading English F5-TTS Base model...")
f5 = F5TTS()
f5_pkg = Path(f5_tts.__path__[0])

# Mother's voice: LJSpeech LJ001-0001 (Linda Johnson) — female, public domain,
# cleanly recorded, 9.66s @22050Hz mono = ideal for F5. Lives in the repo root,
# available under /app via volume mount (.:/app).
# IMPORTANT: REF_TEXT MUST be set. REF_TEXT="" forces F5 to transcribe the
# reference via Whisper ASR — that path loads audio through torchcodec, which
# is broken in the tts image (libavutil/ffmpeg missing) → crash.
# With an explicit transcript, ASR is skipped entirely (= the proven path).
# Text is the exact LJ001-0001 transcript from the LJSpeech metadata.
mother_voice = Path("/app/mother_voice.wav")
if mother_voice.exists():
    REF_AUDIO = str(mother_voice)
    REF_TEXT = (
        "Printing, in the only sense with which we are at present "
        "concerned, differs from most if not from all the arts and "
        "crafts represented in the Exhibition"
    )
    print("Voice: mother_voice.wav (LJSpeech, female)")
else:
    REF_AUDIO = str(f5_pkg / "infer" / "examples" / "basic" / "basic_ref_en.wav")
    REF_TEXT = "Some call me nature, others call me mother nature."
    print("Voice: F5 default male (mother_voice.wav not found)")
print("F5-TTS (English) ready.")

print(f"Reference audio: {REF_AUDIO}")


class SynthRequest(BaseModel):
    text: str


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    async def generate():
        loop = asyncio.get_running_loop()
        wav, sr, _ = await loop.run_in_executor(
            None,
            lambda: f5.infer(
                ref_file=REF_AUDIO,
                ref_text=REF_TEXT,
                gen_text=req.text,
                nfe_step=TTS_NFE_STEP,
            ),
        )
        audio_np = np.clip(wav, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        yield audio_int16.tobytes()

    return StreamingResponse(generate(), media_type="application/octet-stream")
