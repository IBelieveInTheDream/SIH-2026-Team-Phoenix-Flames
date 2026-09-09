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

### Why these flags?

- `--workers 1` – only one process; each extra worker multiplies memory
- `--threads 2` – still handles a few concurrent users without more RAM
- `--timeout 120` – first request after cold start can take time (weather API)

## Environment variables (optional)

| Variable         | Default                     | Purpose                          |
|------------------|-----------------------------|----------------------------------|
| `CENTROIDS_FILE` | `district_centroids.xlsx`   | Path to Excel                    |
| `GEOJSON_FILE`   | `India-Districts-slim.json` | Path to slim GeoJSON             |
| `PORT`           | set by Render               | already handled in app.py        |

## Expected memory after these changes

| Item                         | Approx. RAM   |
|------------------------------|---------------|
| Python + libraries           | 150–250 MB    |
| Slim GeoJSON in memory       | ~5–10 MB      |
| Centroids + weather DataFrame| < 5 MB        |
| Dash / Plotly runtime        | 50–100 MB     |
| **Total (typical)**          | **~250–400 MB** |

This should stay comfortably under the 512 MB hard limit on Render free/starter instances.

## Cold-start note

On free tier the service spins down after inactivity. The first request after wake-up will:

1. Load the slim GeoJSON + Excel
2. Call Open-Meteo for all 641 districts (batched)

This can take 30–90 seconds. The `--timeout 120` prevents gunicorn from killing the process.

## Regenerating the slim GeoJSON later

If you ever update the source districts file:

```bash
python precompute_centroids.py          # creates Excel + full join_key GeoJSON
# then re-run the simplification (or keep the existing India-Districts-slim.json)
```
