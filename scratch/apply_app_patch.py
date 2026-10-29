with open('backend/app.py', 'r', encoding='utf-8') as f:
    c = f.read()

target_req = """class LandmarkRequest(BaseModel):
    landmarks: Optional[List[List[float]]] = None
    multi_hands: Optional[List[dict]] = None
    handedness: Optional[str] = "Right"
    mode: Optional[str] = None"""

replacement_req = """class LandmarkRequest(BaseModel):
    landmarks: Optional[List[List[float]]] = None
    multi_hands: Optional[List[dict]] = None
    handedness: Optional[str] = "Right"
    mode: Optional[str] = None
    timestamp: Optional[float] = None"""

assert target_req in c, "target_req not found"
c = c.replace(target_req, replacement_req, 1)

target_call = """        result = recognizer.predict_landmarks(
            landmarks_list=req.landmarks or [],
            multi_hands=req.multi_hands,
            expected_sign=expected_sign,
            session_id=session_id,
            mode=mode
        )"""

replacement_call = """        result = recognizer.predict_landmarks(
            landmarks_list=req.landmarks or [],
            multi_hands=req.multi_hands,
            expected_sign=expected_sign,
            session_id=session_id,
            mode=mode,
            timestamp=req.timestamp
        )"""

assert target_call in c, "target_call not found"
c = c.replace(target_call, replacement_call, 1)

with open('backend/app.py', 'w', encoding='utf-8') as f:
    f.write(c)

print("backend/app.py updated successfully")

