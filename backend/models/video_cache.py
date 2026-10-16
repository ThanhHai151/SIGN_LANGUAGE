"""
Bộ đệm video/ảnh mẫu QIPEDC trên máy chạy backend
=================================================

Vì sao cần: video gốc của QIPEDC
  • đặt khối chỉ mục "moov" ở CUỐI file → trình duyệt phải tải gần hết file (hoặc nhảy qua lại nhiều lần)
    mới phát được khung đầu tiên → giật, chờ lâu;
  • máy chủ trả "Cache-Control: max-age=0" → mỗi lần mở lại đều tải lại từ đầu.

Backend tải video MỘT lần, chuyển khối "moov" lên đầu (faststart, không mã hóa lại hình ảnh),
lưu vào backend/data/media_cache/ rồi phát từ localhost kèm hỗ trợ tua (HTTP Range) và cache dài hạn.
"""
import os
import re
import struct
import http.client
import ssl
import threading
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(ROOT, "data", "media_cache")
SOURCE = "https://qipedc.moet.gov.vn"
KINDS = {"videos": ".mp4", "thumbs": ".png"}
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
VIDEO_URL_RE = re.compile(r"qipedc\.moet\.gov\.vn/videos/([A-Za-z0-9_-]+)\.mp4")

_locks = {}
_locks_guard = threading.Lock()


def _lock_for(key):
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


# ------------------------------------------------------------------ faststart (đưa moov lên đầu)
def _atoms(buf, start, end):
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[i:i + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[i + 8:i + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - i
        if size < hdr or i + size > end:
            raise ValueError("atom hỏng")
        yield typ, i, size, hdr
        i += size


_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf"}


def _patch_offsets(moov, delta):
    """Cộng delta vào mọi chunk offset (stco/co64) trong moov."""
    out = bytearray(moov)

    def walk(start, end):
        for typ, pos, size, hdr in _atoms(out, start, end):
            body = pos + hdr
            if typ in _CONTAINERS:
                walk(body, pos + size)
            elif typ == b"stco":
                n = struct.unpack(">I", out[body + 4:body + 8])[0]
                for k in range(n):
                    p = body + 8 + 4 * k
                    v = struct.unpack(">I", out[p:p + 4])[0] + delta
                    if v >= 2 ** 32:
                        raise ValueError("offset tràn")
                    out[p:p + 4] = struct.pack(">I", v)
            elif typ == b"co64":
                n = struct.unpack(">I", out[body + 4:body + 8])[0]
                for k in range(n):
                    p = body + 8 + 8 * k
                    v = struct.unpack(">Q", out[p:p + 8])[0] + delta
                    out[p:p + 8] = struct.pack(">Q", v)

    _, _, size, hdr = next(_atoms(out, 0, len(out)))
    walk(hdr, size)
    return bytes(out)


def faststart(data: bytes) -> bytes:
    """Trả về file MP4 có moov đứng trước mdat (giữ nguyên nếu đã đúng hoặc không xử lý được)."""
    try:
        atoms = list(_atoms(data, 0, len(data)))
        types = [a[0] for a in atoms]
        if b"moov" not in types or b"mdat" not in types:
            return data
        mi, di = types.index(b"moov"), types.index(b"mdat")
        if mi < di:
            return data
        _, mpos, msize, _ = atoms[mi]
        moov = _patch_offsets(data[mpos:mpos + msize], msize)
        parts = []
        for k, (typ, pos, size, _) in enumerate(atoms):
            if k == mi:
                continue
            if k == di:
                parts.append(moov)
            parts.append(data[pos:pos + size])
        out = b"".join(parts)
        return out if len(out) == len(data) else data
    except (ValueError, struct.error, StopIteration):
        return data


# ------------------------------------------------------------------ tải & lưu
_tls = threading.local()
HOST = "qipedc.moet.gov.vn"


def _conn(timeout, verify=True):
    """Kết nối HTTPS giữ mở (keep-alive) cho mỗi luồng: bỏ được ~1 giây DNS + TCP + TLS cho mỗi video."""
    key = "c_v" if verify else "c_u"
    c = getattr(_tls, key, None)
    if c is None:
        ctx = ssl.create_default_context() if verify else ssl._create_unverified_context()  # noqa: S323
        c = http.client.HTTPSConnection(HOST, 443, timeout=timeout, context=ctx)
        setattr(_tls, key, c)
    return c


def _drop(verify):
    key = "c_v" if verify else "c_u"
    c = getattr(_tls, key, None)
    if c is not None:
        try:
            c.close()
        except Exception:  # noqa: BLE001
            pass
        setattr(_tls, key, None)


def _get(path, timeout, verify):
    for attempt in range(2):
        c = _conn(timeout, verify)
        try:
            c.request("GET", path, headers={"User-Agent": "Mozilla/5.0 (VSL Connect AI)", "Connection": "keep-alive"})
            r = c.getresponse()
            data = r.read()
            if r.status != 200:
                raise OSError(f"HTTP {r.status}")
            return data
        except ssl.SSLError:
            _drop(verify)
            raise
        except (http.client.HTTPException, OSError):
            _drop(verify)  # kết nối cũ bị máy chủ đóng → mở lại 1 lần
            if attempt:
                raise
    return b""


def _download(path, timeout):
    try:
        return _get(path, timeout, True)
    except ssl.SSLError:
        # Một số máy Windows thiếu chứng chỉ trung gian của máy chủ QIPEDC → thử lại không kiểm tra chứng chỉ
        # (chỉ áp dụng cho video/ảnh công khai, chỉ đọc).
        return _get(path, timeout, False)


def cache_path(kind, media_id):
    return os.path.join(CACHE_DIR, kind, media_id + KINDS[kind])


def ensure_cached(kind, media_id, timeout=20):
    """Đảm bảo file đã có trong bộ đệm; trả về đường dẫn hoặc None nếu không tải được."""
    if kind not in KINDS or not ID_RE.match(media_id or ""):
        return None
    path = cache_path(kind, media_id)
    if os.path.exists(path):
        return path
    with _lock_for((kind, media_id)):
        if os.path.exists(path):
            return path
        try:
            data = _download(f"/{kind}/{media_id}{KINDS[kind]}", timeout)
        except Exception:  # noqa: BLE001 — mạng lỗi → để frontend dùng link gốc
            return None
        if kind == "videos":
            if len(data) < 1000:
                return None
            data = faststart(data)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return path


_queue = []                 # hàng đợi tải trước (mới nhất được ưu tiên)
_queued = set()
_queue_cv = threading.Condition()
_workers = []
N_WORKERS = 3


def _worker():
    while True:
        with _queue_cv:
            while not _queue:
                _queue_cv.wait()
            kind, mid = _queue.pop()   # LIFO: ô người dùng vừa lướt tới được tải trước
            _queued.discard((kind, mid))
        try:
            ensure_cached(kind, mid)
        except Exception:  # noqa: BLE001
            pass


def _start_workers():
    if not _workers:
        for _ in range(N_WORKERS):
            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            _workers.append(t)


def warm_background(kind, ids):
    """Tải dần TOÀN BỘ danh sách ở nền với độ ưu tiên thấp nhất (đứng sau các yêu cầu từ giao diện)."""
    with _queue_cv:
        _start_workers()
        todo = [(kind, m) for m in ids if ID_RE.match(m or "") and (kind, m) not in _queued and not os.path.exists(cache_path(kind, m))]
        # đáy hàng đợi LIFO = tải sau cùng; giữ thứ tự: phần tử đầu danh sách được tải trước
        _queue[:0] = list(reversed(todo))
        _queued.update(todo)
        _queue_cv.notify_all()
    return len(todo)


def prefetch(kind, ids):
    """Tải trước ở nền bằng 4 luồng cố định (giữ kết nối keep-alive) để bấm xem là phát ngay."""
    with _queue_cv:
        _start_workers()
        n = 0
        for mid in reversed(ids[:60]):   # đảo để ô đầu danh sách nằm cuối hàng đợi LIFO → tải trước
            key = (kind, mid)
            if ID_RE.match(mid or "") and key not in _queued and not os.path.exists(cache_path(kind, mid)):
                _queue.append(key)
                _queued.add(key)
                n += 1
        _queue_cv.notify_all()
    return n


def read_range(path, range_header):
    """Đọc toàn bộ hoặc một đoạn (HTTP Range) của file. Trả (status, data, headers)."""
    size = os.path.getsize(path)
    start, end = 0, size - 1
    status = 200
    m = re.match(r"bytes=(\d*)-(\d*)", range_header or "")
    if m and (m.group(1) or m.group(2)):
        if m.group(1):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
        else:
            start = max(0, size - int(m.group(2)))
        end = min(end, size - 1)
        if start > end:
            return 416, b"", {"Content-Range": f"bytes */{size}"}
        status = 206
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start + 1)
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=31536000, immutable"}
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return status, data, headers


def diagnose(media_id="W00489"):
    """Đo từng bước tải 1 video từ QIPEDC (DNS, kết nối, TLS, tải) — để tìm nguyên nhân chậm."""
    import socket
    import time
    host = "qipedc.moet.gov.vn"
    res = {}
    t = time.perf_counter()
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        res["dns_ms"] = round((time.perf_counter() - t) * 1000)
        res["addrs"] = sorted({f"{'v6' if i[0] == socket.AF_INET6 else 'v4'} {i[4][0]}" for i in infos})
    except OSError as e:
        res["dns_error"] = str(e)
        return res
    for fam, _, _, _, addr in infos[:4]:
        t = time.perf_counter()
        try:
            s = socket.socket(fam, socket.SOCK_STREAM)
            s.settimeout(8)
            s.connect(addr)
            s.close()
            res[f"connect {addr[0]}"] = round((time.perf_counter() - t) * 1000)
        except OSError as e:
            res[f"connect {addr[0]}"] = f"lỗi sau {round((time.perf_counter() - t) * 1000)} ms: {e}"
    for label, ctx in (("verified", ssl.create_default_context()), ("unverified", ssl._create_unverified_context())):  # noqa: S323
        t = time.perf_counter()
        try:
            req = urllib.request.Request(f"{SOURCE}/videos/{media_id}.mp4", headers={"User-Agent": "Mozilla/5.0 (VSL Connect AI)"})
            with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
                ttfb = time.perf_counter() - t
                n = len(r.read())
            res[f"get_{label}"] = {"ttfb_ms": round(ttfb * 1000), "total_ms": round((time.perf_counter() - t) * 1000), "bytes": n}
        except Exception as e:  # noqa: BLE001
            res[f"get_{label}"] = f"lỗi sau {round((time.perf_counter() - t) * 1000)} ms: {type(e).__name__}: {e}"
    # đường tải thực tế của bộ đệm (kết nối keep-alive) + bước faststart
    for k in range(2):
        t = time.perf_counter()
        try:
            d = _download(f"/videos/{media_id}.mp4", 20)
            t1 = time.perf_counter()
            faststart(d)
            res[f"keepalive_get_{k}"] = {"get_ms": round((t1 - t) * 1000), "faststart_ms": round((time.perf_counter() - t1) * 1000)}
        except Exception as e:  # noqa: BLE001
            res[f"keepalive_get_{k}"] = f"{type(e).__name__}: {e}"
    with _queue_cv:
        res["prefetch_queue"] = len(_queue)
    import os as _os
    res["proxy_env"] = {k: v for k, v in _os.environ.items() if "proxy" in k.lower()}
    return res
