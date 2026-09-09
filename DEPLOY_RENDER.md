# Deploy to Render (512 MB limit)

## Files you must include in the repo / deploy directory

```
app.py
district_centroids.xlsx
India-Districts-slim.json      ← use the slim one (2.7 MB)
requirements.txt
```

Do **not** upload the original 26 MB GeoJSON.

## Render service settings

1. **New → Web Service**
2. Connect your Git repo (or use a Blueprint)
3. Settings:

| Setting              | Value                                      |
|----------------------|--------------------------------------------|
| Runtime              | Python 3                                   |
| Build Command        | `pip install -r requirements.txt`          |
| Start Command        | `gunicorn app:server --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT` |
| Instance Type        | Free or Starter (512 MB)                   |

### Critical: Start Command (fixes "No open ports detected")

Copy this **exactly**:

```
gunicorn app:server --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT
```

Common mistakes that cause **Port scan timeout / No open ports**:
- Using `python app.py` instead of gunicorn
- Forgetting `--bind 0.0.0.0:$PORT`
- Using more than 1 worker on free tier (OOM → process dies before binding)

### Why these flags?

- `--workers 1` – only one process; each extra worker multiplies memory
- `--threads 2` – handles a few concurrent users without more RAM
- `--timeout 120` – background weather refresh can take a while
- `--bind 0.0.0.0:$PORT` – **required** so Render's health check finds the port

## Environment variables (optional)

| Variable         | Default                     | Purpose                          |
|------------------|-----------------------------|----------------------------------|
| `CENTROIDS_FILE` | `district_centroids.xlsx`   | Path to Excel                    |
| `GEOJSON_FILE`   | `India-Districts-slim.json` | Path to slim GeoJSON             |
| `PORT`           | set by Render               | already handled by gunicorn      |

## Expected memory after these changes

| Item                         | Approx. RAM   |
|------------------------------|---------------|
| Python + libraries           | 150–250 MB    |
| Slim GeoJSON in memory       | ~5–10 MB      |
| Centroids + weather DataFrame| < 5 MB        |
| Dash / Plotly runtime        | 50–100 MB     |
| **Total (typical)**          | **~250–400 MB** |

This should stay under the 512 MB hard limit.

## How startup works now (avoids port timeout)

1. Process starts → loads Excel + slim GeoJSON (fast, ~1–3 s)
2. Fills **synthetic** weather data instantly
3. **Binds to $PORT immediately** → Render health check succeeds
4. Background thread fetches real Open-Meteo data and swaps it in

You no longer wait for the weather API before the port opens.

## Regenerating the slim GeoJSON later

If you ever update the source districts file:

```bash
python precompute_centroids.py          # creates Excel + full join_key GeoJSON
# then re-run the simplification (or keep the existing India-Districts-slim.json)
```
