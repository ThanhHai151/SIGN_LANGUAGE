import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

urls = [
    "https://qipedc.moet.gov.vn/videos/D0507N.mp4",
    "https://qipedc.moet.gov.vn/videos/D0507T.mp4",
    "https://qipedc.moet.gov.vn/videos/D0507B.mp4",
    "https://qipedc.moet.gov.vn/thumbs/D0507N.png",
    "https://qipedc.moet.gov.vn/thumbs/D0507B.png"
]

for url in urls:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            cl = resp.headers.get("Content-Length")
            ct = resp.headers.get("Content-Type")
            print(f"{url} -> Status: {resp.status}, Content-Length: {cl}, Content-Type: {ct}")
    except Exception as e:
        print(f"{url} -> Error: {e}")
