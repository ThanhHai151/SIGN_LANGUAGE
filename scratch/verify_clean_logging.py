import os
import shutil
import csv
import json
import numpy as np

from backend.vision.word_debugger import word_debugger, MASTER_CSV_PATH, CSV_FIELDNAMES

def main():
    print("=== TEST START: VERIFY CLEAN WEBCAM LOGGING ENVIRONMENT ===")
    
    # 0. Ensure clean starting state for master CSV
    with open(MASTER_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
    res = word_debugger.start_session(
        target_class=39,
        target_word="Thời gian",
        trial=1,
        checkpoint_info={
            "checkpoint_type": "historical",
            "checkpoint_filename": "sign_sstcn_epoch_32.pth",
            "checkpoint_sha256": "abcdef1234567890",
            "checkpoint_epoch": 32,
            "model_device": "cpu"
        },
        session_prefix="A01_1_T01"
    )
    
    assert res["success"] is True, f"Failed to start session: {res.get('message')}"
    session_id = res["session_id"]
    print(f"[OK] Session started successfully: {session_id}")
    assert session_id.startswith("A01_1_T01_Thoi_gian_trial_01_"), f"Unexpected session_id: {session_id}"
    
    session_dir = word_debugger.session_dir
    assert os.path.isdir(session_dir), f"Session directory not created: {session_dir}"
    assert os.path.isfile(os.path.join(session_dir, "session.json")), "session.json missing"
    assert os.path.isfile(os.path.join(session_dir, "events.jsonl")), "events.jsonl missing"
    
    # Check session.json contents
    with open(os.path.join(session_dir, "session.json"), "r", encoding="utf-8") as f:
        sess_data = json.load(f)
    assert sess_data.get("campaign") == "A01_1_T01", f"Campaign not saved in session.json: {sess_data.get('campaign')}"
    print(f"[OK] session.json verified with campaign: {sess_data.get('campaign')}")
    
    # 2. Simulate recording a sequence
    dummy_input = np.random.randn(30, 75, 3).astype(np.float32)
    top_k = [{"index": 39, "term": "Thời gian", "confidence": 0.95}]
    word_debugger.record_prediction(
        arr_input=dummy_input,
        top_k_list=top_k,
        max_prob=0.95,
        pred_term="Thời gian",
        threshold=0.5,
        inference_latency_ms=12.5,
        debug_meta={"tracker_engine": "holistic", "active_instance_engine": "holistic", "client_fps": 30.0}
    )
    
    # Check status
    st = word_debugger.get_status()
    assert st["seq_count"] == 1, f"Expected 1 sequence, got {st['seq_count']}"
    assert st["campaign"] == "A01_1_T01", f"Expected campaign A01_1_T01, got {st['campaign']}"
    print(f"[OK] Sequence recorded. Status: {st}")
    
    # 3. Stop session
    stop_res = word_debugger.stop_session()
    assert stop_res["success"] is True, "Failed to stop session"
    summary = stop_res["summary"]
    assert summary["verdict"] == "CORRECT", f"Expected CORRECT verdict, got {summary['verdict']}"
    assert summary["total_windows"] == 1, f"Expected 1 window, got {summary['total_windows']}"
    print(f"[OK] Session stopped with summary verdict: {summary['verdict']}")
    
    # 4. Check master CSV
    with open(MASTER_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1, f"Expected 1 row in master CSV, got {len(rows)}"
    assert rows[0]["session_id"] == session_id, f"Master CSV session_id mismatch: {rows[0]['session_id']}"
    print(f"[OK] Master CSV verified: 1 row recorded with session_id={rows[0]['session_id']}")
    
    # 5. Verify archive directory is intact
    archive_dir = os.path.join(os.path.dirname(os.path.dirname(session_dir)), "archive", "vsl_p010_historical_reconstructed_2026-11-07")
    assert os.path.isdir(archive_dir), f"Archive dir does not exist: {archive_dir}"
    with open(os.path.join(archive_dir, "ARCHIVE_MANIFEST.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["total_copied_files"] == 544, f"Archive file count corrupted: {manifest.get('total_copied_files')}"
    print(f"[OK] Archive verification passed: {manifest['total_copied_files']} files intact.")
    
    # 6. Cleanup test session folder and reset trials_summary.csv so directory is pristine for real testing
    shutil.rmtree(session_dir)
    with open(MASTER_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
    print("[OK] Test session cleaned up. Clean environment ready for user's 15 trials!")
    print("=== ALL VERIFICATION CHECKS PASSED ===")

if __name__ == "__main__":
    main()
