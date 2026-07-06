import json, re, urllib.request
url = "https://www.pexels.com/video/27727783/"
html = urllib.request.urlopen(
    urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60
).read().decode("utf-8", "replace")
title = re.search(r"<title>([^<]+)</title>", html)
print("title:", title.group(1).strip() if title else "?")
for block in re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S):
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        continue
    if isinstance(data, dict) and data.get("@type") == "VideoObject":
        print("name:", data.get("name"))
        print("author:", data.get("author"))
        print("description:", (data.get("description") or "")[:200])
