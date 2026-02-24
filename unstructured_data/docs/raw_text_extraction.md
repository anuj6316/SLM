```
Start
  │
  ▼
Initialize visited set (if None)
  │
  ▼
Determine current_url:
  current_url = cfg.url if cfg.active_url is None else cfg.active_url
  │
  ▼
Is current_url in visited?
  ├─ Yes → Log "Skipping current_url" → Return
  └─ No  → Add current_url to visited
              │
              ▼
          Build request URL:
          url = "https://r.jina.ai/" + current_url
              │
              ▼
        Send GET request with headers (API key)
              │
              ▼
        Log response status & URL
              │
              ▼
        Write response text to "output.md"
              │
              ▼
        Extract links from response text
        ┌───────────────────────────────┐
        │ Only links starting with      │
        │ https://www.mindmapdigital.ai│
        │ and without '#' are kept      │
        └───────────────────────────────┘
              │
              ▼
    For each extracted link:
          ┌─────────────────────────────┐
          │ If link not in visited      │
          │   cfg.active_url = link     │
          │   Recurse → get_raw_content │
          └─────────────────────────────┘
              │
              ▼
         All links processed
              │
              ▼
             End
```