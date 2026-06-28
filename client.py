import os
import subprocess
import sys

def _selfboot():
    import shutil, signal
    here = os.path.dirname(os.path.abspath(__file__))
    venv = os.path.join(here, ".venv")
    py   = os.path.join(venv, "Scripts" if os.name == "nt" else "bin", "python.exe" if os.name == "nt" else "python")
    if not os.path.exists(py):
        print("Setting up environment (first time only, ~1 min)...")
        # Subprocesses in their own process group — SIGINT from ensurepip
        # cannot reach the SSH process (= server connection).
        _pg = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" \
              else {"preexec_fn": os.setsid}
        old = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            for attempt in range(2):
                shutil.rmtree(venv, ignore_errors=True)
                try:
                    subprocess.check_call([sys.executable, "-m", "venv", "--without-pip", venv], **_pg)
                    subprocess.check_call([py, "-m", "ensurepip", "--upgrade"], **_pg)
                    break
                except Exception:
                    if attempt == 1:
                        raise
            subprocess.check_call([py, "-m", "pip", "install",
                                   "websockets", "sounddevice", "numpy", "PySide6"], **_pg)
        finally:
            signal.signal(signal.SIGINT, old)
        print("Done.")
    if os.path.abspath(sys.executable) != os.path.abspath(py):
        if os.name == "nt":
            sys.exit(subprocess.call([py] + sys.argv))
        else:
            os.execv(py, [py] + sys.argv)

_selfboot()

import asyncio
import websockets
import sounddevice as sd
import numpy as np
import config

mother_speaking = False
session_ended = False   # Server sent SESSION_END → exit, do NOT reconnect

# ── Status hook for the optional GUI ──────────────────────────────────────
# In terminal mode (UX_EXPERIENCE=False) _status_hook stays None → all
# publish_status calls are no-ops, behaviour identical to before. If the GUI
# sets a hook, it receives every state change as (state, detail).
_status_hook = None

def set_status_hook(fn):
    global _status_hook
    _status_hook = fn

def publish_status(state, detail=""):
    if _status_hook is not None:
        _status_hook(state, detail)

SERVER_HOST = config.SERVER_HOST
SERVER_PORT = config.SERVER_PORT

SAMPLE_RATE = config.SAMPLE_RATE
CHUNK_DURATION = config.CHUNK_DURATION
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # = 1600 samples

def get_audio_chunk(indata):
    chunk = indata[:, 0]  # mono
    return chunk.astype(np.float32).tobytes()  # numpy array → bytes

async def stream_audio():
    uri = f"ws://{SERVER_HOST}:{SERVER_PORT}"

    while True:
        try:
            publish_status("connecting", uri)
            print(f"Connecting to {uri}...")
            # ping_interval=None: no keepalive kill — mirrors the server side
            # (server.py serve(..., ping_interval=None)). While Mother is speaking
            # (long answers = 30-40s audio) the client is busy with blocking
            # stream.write() and cannot respond to the 20s default ping-pong
            # in time → the connection would drop after EVERY long answer.
            # Real network drops are still caught by ConnectionClosed/OSError
            # below → reconnect remains intact.
            async with websockets.connect(uri, ping_interval=None) as websocket:
                publish_status("listening")
                print("Connected — microphone running (Ctrl+C to stop)")
                print("-" * 50)

                # Audio queue for communication between sounddevice thread and asyncio
                queue = asyncio.Queue()
                loop = asyncio.get_event_loop()

                def callback(indata, frames, time, status):
                    chunk_bytes = get_audio_chunk(indata)
                    loop.call_soon_threadsafe(queue.put_nowait, chunk_bytes)

                async def send_loop():
                    global mother_speaking
                    while True:
                        chunk_bytes = await queue.get()
                        while not queue.empty():
                            chunk_bytes = queue.get_nowait()
                        if not mother_speaking:
                            await websocket.send(chunk_bytes)

                async def receive_loop():
                    global mother_speaking, session_ended
                    stream = None
                    byte_buffer = b""
                    try:
                        async for message in websocket:
                            if isinstance(message, bytes):
                                if stream is None:
                                    mother_speaking = True
                                    publish_status("speaking")
                                    stream = sd.RawOutputStream(samplerate=config.TTS_SAMPLE_RATE, channels=1, dtype='int16')
                                    stream.start()
                                byte_buffer += message
                                write_len = len(byte_buffer) - (len(byte_buffer) % 2)
                                if write_len > 0:
                                    stream.write(byte_buffer[:write_len])
                                    byte_buffer = byte_buffer[write_len:]
                            elif message == "BUSY":
                                # Server detected silence / pipeline triggered →
                                # mute immediately, covers the entire processing window
                                mother_speaking = True
                                publish_status("thinking")
                            elif message == "TTS_DONE":
                                if stream is not None:
                                    stream.stop()
                                    stream.close()
                                    stream = None
                                byte_buffer = b""
                                # Hangover against self-triggering without headphones: speaker +
                                # room reverb still linger. Keep mic muted, then discard audio
                                # captured in the meantime so Mother's last word does not arrive
                                # as a user turn.
                                await asyncio.sleep(config.MIC_REARM_COOLDOWN)
                                while not queue.empty():
                                    queue.get_nowait()
                                mother_speaking = False
                                publish_status("listening")
                            elif message == "SESSION_END":
                                # Server ends the session (kill phrase). Play remaining audio,
                                # then close connection → outer loop exits (no reconnect) → script
                                # ends → run_session.sh trap pulls the results.
                                if stream is not None:
                                    stream.stop()
                                    stream.close()
                                    stream = None
                                session_ended = True
                                publish_status("ended")
                                print("Session ended by server.")
                                await websocket.close()
                                return
                    finally:
                        mother_speaking = False
                        if stream is not None:
                            stream.stop()
                            stream.close()

                with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE, callback=callback):
                    await asyncio.gather(send_loop(), receive_loop())

        except websockets.exceptions.ConnectionClosed:
            if session_ended:
                break
            publish_status("reconnecting")
            print("Connection lost — trying to reconnect...")
        except OSError:
            publish_status("reconnecting")
            print("Server unreachable — trying to reconnect...")

        if session_ended:
            break

        await asyncio.sleep(config.RECONNECT_DELAY)

    publish_status("stopped")
    print("Client stopped.")


def _sync_distillates(username):
    import subprocess, time, os
    server = f"{username}@10.28.18.6"
    remote_dir = "~/ourBr00d/distillates"
    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "distillates")

    print("\nSession ended — waiting for distillation on server...")

    max_wait = 40
    elapsed = 0
    done = False
    while elapsed < max_wait:
        try:
            latest = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", server,
                 f"ls -t {remote_dir}/sessions/session_*.txt 2>/dev/null | head -1"],
                capture_output=True, text=True, timeout=10
            ).stdout.strip()
            if latest:
                marker = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", server, f"tail -3 '{latest}'"],
                    capture_output=True, text=True, timeout=10
                ).stdout
                if "distillation:" in marker:
                    done = True
                    break
        except Exception:
            pass
        time.sleep(2)
        elapsed += 2

    print("Distillation complete — syncing..." if done else f"Timeout after {max_wait}s — syncing anyway...")
    os.makedirs(local_dir, exist_ok=True)
    if os.name == "nt":
        # rsync not available on Windows — use scp (built into Windows 10+ via OpenSSH)
        subprocess.run(["scp", "-r", f"{server}:{remote_dir}/sessions", local_dir])
        subprocess.run(["scp", f"{server}:{remote_dir}/lessons.md", local_dir])
    else:
        subprocess.run(["rsync", "-avz", f"{server}:{remote_dir}/", local_dir + "/"])
    print(f"Sync done. Transcripts at: {local_dir}")


def main():
    # Username as argument: python client.py dilan
    # No argument → prompt as fallback
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        print("Your server login: ssh [USERNAME]@10.28.18.6")
        username = input("Enter your USERNAME here: ").strip()

    # UX_EXPERIENCE controls display purely on the client side.
    # False → terminal as before. True → GUI takes the main thread and
    # runs stream_audio() in a daemon thread, registers the status hook
    # and mirrors every state change to the window.
    if config.UX_EXPERIENCE:
        import gui
        gui.run(stream_audio, set_status_hook)
    else:
        asyncio.run(stream_audio())

    if session_ended:
        _sync_distillates(username)


if __name__ == "__main__":
    main()
