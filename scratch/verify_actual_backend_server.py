import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error

def run_backend_verification(python_exe, env_name, port=8000):
    print(f"\n=======================================================")
    print(f"VERIFYING BACKEND IN ENVIRONMENT: {env_name}")
    print(f"Python: {python_exe}")
    print(f"Port: {port}")
    print(f"=======================================================")

    # 1. Print python and pytorch version
    v_proc = subprocess.run([python_exe, "-c", "import sys, torch; print('PY_VER:', sys.version); print('TORCH_VER:', torch.__version__)"], capture_output=True, text=True)
    print("Environment info:\n" + v_proc.stdout.strip())

    # 2. Start Uvicorn backend process with file log to prevent pipe buffer deadlock
    log_out_path = os.path.abspath(f"scratch/server_uvicorn_{port}.log")
    log_file = open(log_out_path, "w", encoding="utf-8")

    cmd = [python_exe, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", str(port)]
    server_proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=os.path.abspath("."))

    # 3. Wait for server to become responsive
    url_health = f"http://127.0.0.1:{port}/api/health"
    url_control = f"http://127.0.0.1:{port}/api/pilot/control"
    url_recognize = f"http://127.0.0.1:{port}/api/recognize_landmarks"
    url_flush = f"http://127.0.0.1:{port}/api/pilot/flush"

    server_ready = False
    for attempt in range(40):
        time.sleep(0.5)
        try:
            req = urllib.request.Request(url_health)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    health_data = json.loads(resp.read().decode("utf-8"))
                    print(f"Server is online! (Attempt {attempt+1}) Health: {health_data['status']}")
                    server_ready = True
                    break
        except Exception:
            if server_proc.poll() is not None:
                print("Server exited prematurely!")
                break

    if not server_ready:
        print("Failed to reach server. Checking log file:")
        log_file.flush()
        with open(log_out_path, "r", encoding="utf-8", errors="replace") as f:
            print(f.read())
        server_proc.kill()
        log_file.close()
        return None

    session_id = f"verify_{env_name}_{int(time.time())}"
    # 4. Enable pilot logging to record live recognizer payload
    ctrl_payload = json.dumps({"enabled": True, "session_id": session_id}).encode("utf-8")
    req_ctrl = urllib.request.Request(url_control, data=ctrl_payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req_ctrl) as resp:
        ctrl_res = json.loads(resp.read().decode("utf-8"))
    log_filepath = ctrl_res.get("filepath")
    print(f"Pilot session enabled. Log file: {log_filepath}")

    # 5. Send POST /api/recognize_landmarks
    fixture_path = os.path.join("backend", "tests", "fixtures", "real_alphabet_landmarks.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        fixture_data = json.load(f)
    a_sample = next(s for s in fixture_data["samples"] if s["letter_upper"] == "A")

    rec_payload = json.dumps({
        "landmarks": a_sample["landmarks"],
        "mode": "letters",
        "expected_sign": "A",
        "session_id": session_id,
        "trial_id": f"{session_id}_A_rep01_att01",
        "frame_idx": 1,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "X-Conversion-Mode": "letters",
        "X-Expected-Sign": "A",
        "X-Session-Id": session_id,
        "X-Trial-Id": f"{session_id}_A_rep01_att01",
        "X-Frame-Idx": "1",
    }
    req_rec = urllib.request.Request(url_recognize, data=rec_payload, headers=headers)
    with urllib.request.urlopen(req_rec) as resp:
        rec_res = json.loads(resp.read().decode("utf-8"))
    print(f"API Response from POST /api/recognize_landmarks:\n  {json.dumps(rec_res, ensure_ascii=False)}")

    # 6. Flush pilot logger
    req_flush = urllib.request.Request(url_flush, data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req_flush) as resp:
        flush_res = json.loads(resp.read().decode("utf-8"))
    print(f"Flush status: {flush_res}")

    # 7. Disable pilot logger
    ctrl_disable = json.dumps({"enabled": False}).encode("utf-8")
    req_disable = urllib.request.Request(url_control, data=ctrl_disable, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req_disable) as resp:
        pass

    # 8. Terminate server process cleanly
    server_proc.terminate()
    try:
        server_proc.wait(timeout=5)
    except Exception:
        server_proc.kill()
        server_proc.wait()
    log_file.close()

    print("\nServer startup log snippet (filtered for Static Alphabet Head):")
    with open(log_out_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if any(kw in line for kw in ["static", "head", "head_v1", "alphabet", "Warning", "Weights", "UnpicklingError", "VSLRecognizer", "Uvicorn running"]):
                print("  ", line.strip())

    # 9. Read the pilot record from disk
    record = None
    if log_filepath and os.path.exists(log_filepath):
        with open(log_filepath, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    record = json.loads(l)
                    break
        try:
            os.remove(log_filepath)
        except Exception:
            pass

    if record:
        print("\nRecorded Pilot Frame Details:")
        print(f"  has_static_head:     {record.get('has_static_head')}")
        print(f"  static_head info:    {json.dumps(record.get('static_head'), ensure_ascii=False)}")
        print(f"  final_sign:          {record.get('final_sign')}")
        print(f"  prediction_source:   {record.get('prediction_source')}")
    else:
        print("No pilot record found on disk!")

    return record

if __name__ == "__main__":
    # Test Environment 1: .venv
    venv_python = os.path.abspath(os.path.join(".", ".venv", "Scripts", "python.exe"))
    rec_venv = None
    if os.path.exists(venv_python):
        rec_venv = run_backend_verification(venv_python, "venv", port=8001)

    # Test Environment 2: Global Python
    global_python = sys.executable
    rec_global = run_backend_verification(global_python, "Global_Python", port=8002)

    print("\n=======================================================")
    print("FINAL SUMMARY COMPARISON")
    print("=======================================================")
    if rec_venv:
        sh = rec_venv.get("static_head", {})
        print(f".venv (Port 8001):")
        print(f"  PyTorch:         {sh.get('pytorch_version')}")
        print(f"  has_static_head: {rec_venv.get('has_static_head')}")
        print(f"  load_status:     {sh.get('load_status')}")
        print(f"  load_error:      {sh.get('load_error')}")
    if rec_global:
        sh_g = rec_global.get("static_head", {})
        print(f"Global Python (Port 8002):")
        print(f"  PyTorch:         {sh_g.get('pytorch_version')}")
        print(f"  has_static_head: {rec_global.get('has_static_head')}")
        print(f"  load_status:     {sh_g.get('load_status')}")
        print(f"  load_error:      {sh_g.get('load_error')}")
