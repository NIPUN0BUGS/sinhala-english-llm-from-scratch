#!/usr/bin/env python3
"""Free-media CI smoke test: Pollinations image + Wikimedia metadata + MP4."""
from __future__ import annotations
import json, os, re, shutil, subprocess, sys, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(os.environ.get("MEDIA_OUTPUT_DIR", "media-output"))
OUT.mkdir(parents=True, exist_ok=True)
PROMPT = os.environ.get("IMAGE_PROMPT", "friendly colorful 2D cartoon scene of a curious little robot exploring a green Sri Lankan garden, clean composition, no text")
COMMONS_QUERY = os.environ.get("COMMONS_QUERY", "animation cartoon")
UA = "ZEE-FreeMedia-CI/1.0 (metadata discovery; contact repository maintainer)"

def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()

def generate_image():
    url = "https://image.pollinations.ai/prompt/" + urllib.parse.quote(PROMPT, safe="") + "?width=1280&height=720&nologo=true&seed=42"
    target = OUT / "generated-image.jpg"
    try:
        data = fetch(url, 90)
        if len(data) < 10000:
            raise ValueError("Image response was unexpectedly small")
        if not (data.startswith(b"\xff\xd8\xff") or data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"RIFF")):
            raise ValueError("Provider response was not a recognized image")
        target.write_bytes(data)
        return {"status":"success","provider":"Pollinations","prompt":PROMPT,"file":target.name,"bytes":len(data),"source_url":url}
    except Exception as exc:
        return {"status":"failed","provider":"Pollinations","prompt":PROMPT,"error":f"{type(exc).__name__}: {exc}","source_url":url}

def discover_commons():
    params = {"action":"query","format":"json","generator":"search","gsrsearch":f"filetype:video {COMMONS_QUERY}","gsrnamespace":"6","gsrlimit":"12","prop":"imageinfo","iiprop":"url|extmetadata"}
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    try:
        raw = json.loads(fetch(url,45).decode("utf-8"))
        assets = []
        for page in raw.get("query",{}).get("pages",{}).values():
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}
            def val(key):
                item = meta.get(key) or {}
                return re.sub("<[^>]+>", "", str(item.get("value",""))).strip()
            title = page.get("title","")
            assets.append({"title":title,"file_page":"https://commons.wikimedia.org/wiki/"+urllib.parse.quote(title.replace(" ","_"),safe=":/"),"original_url":info.get("url",""),"license":val("LicenseShortName"),"license_url":val("LicenseUrl"),"artist":val("Artist"),"credit":val("Credit"),"attribution_required":val("AttributionRequired"),"reuse_note":"Metadata discovery only. Verify the individual file page, commercial-use rights, attribution and ShareAlike obligations before reuse."})
        result = {"status":"success","query":COMMONS_QUERY,"count":len(assets),"assets":assets}
    except Exception as exc:
        result = {"status":"failed","query":COMMONS_QUERY,"count":0,"assets":[],"error":f"{type(exc).__name__}: {exc}"}
    (OUT/"wikimedia-video-candidates.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"status":result["status"],"count":result["count"],"manifest":"wikimedia-video-candidates.json","error":result.get("error")}

def render_mp4(image_result):
    image, video = OUT/"generated-image.jpg", OUT/"zee-demo.mp4"
    if image_result.get("status") != "success" or not image.exists():
        return {"status":"skipped","reason":"No generated image; check provider status in run-report.json"}
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"status":"failed","error":"ffmpeg is not installed"}
    cmd=[ffmpeg,"-y","-loop","1","-i",str(image),"-vf","scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p","-t","6","-r","24","-c:v","libx264","-movflags","+faststart",str(video)]
    try:
        proc=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        if proc.returncode:
            return {"status":"failed","error":proc.stderr[-2000:]}
        return {"status":"success","file":video.name,"bytes":video.stat().st_size,"duration_seconds":6,"note":"Six-second MP4 from one generated still; not a multi-shot video model."}
    except Exception as exc:
        return {"status":"failed","error":f"{type(exc).__name__}: {exc}"}

def main():
    report={"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"policy":{"commons_downloaded":False,"third_party_video_embedded":False,"commercial_reuse_approved":False,"manual_asset_level_license_review_required":True}}
    report["image"]=generate_image()
    report["commons_discovery"]=discover_commons()
    report["video"]=render_mp4(report["image"])
    (OUT/"run-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"image_status":report["image"]["status"],"commons_status":report["commons_discovery"]["status"],"commons_count":report["commons_discovery"]["count"],"video_status":report["video"]["status"],"report":str(OUT/"run-report.json")},indent=2))
    return 0 if (OUT/"run-report.json").exists() else 1

if __name__=="__main__":
    sys.exit(main())
