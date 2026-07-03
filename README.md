# TrackFlow MECHMAXX Label Image Generator

Generates a realistic MECHMAXX package photo with a dynamic `SHIP TO` address and dynamic TrackFlow tracking number.

## Static fields

The generator keeps these parts from `base.png`:

- `SHIP FROM` address
- barcode bars
- package/background/photo perspective

Only these fields change:

- `SHIP TO`
- tracking number text under the barcode

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open:

```txt
http://127.0.0.1:8000/docs
```

## Render settings

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Test body

```json
{
  "recipient": {
    "name": "Steve Cross",
    "line1": "9200 Delashmit Road",
    "line2": "Millington, TN",
    "line3": "38053",
    "line4": "US",
    "phone": ""
  },
  "tracking_number": "TF-QZ6HTDVPRH",
  "response_format": "image"
}
```

Use `"response_format": "base64"` if Lovable needs base64 instead of an image response.
