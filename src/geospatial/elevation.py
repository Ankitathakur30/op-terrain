# src/geospatial/elevation.py
import requests
import numpy as np
import time

def fetch_elevation_grid(lat_min, lat_max, lon_min, lon_max, resolution=10):
    lats = np.linspace(lat_min, lat_max, resolution)   # evenly spaced points across the box
    lons = np.linspace(lon_min, lon_max, resolution)

    # cartesian product of lats x lons = every grid point, flattened
    locations = [(lat,lon) for lat in lats for lon in lons]

    elevations = []
    batch_size = 100  # API's per-request cap

    for i in range(0, len(locations), batch_size):
        batch = locations[i:i + batch_size]
        locations_str = "|".join(f"{lat},{lon}" for lat, lon in batch)

        response = requests.get(
            "https://api.opentopodata.org/v1/srtm90m",
            params={"locations": locations_str},
            timeout=30
        )
        response.raise_for_status()  # fail loud, not silent, on bad response
        results = response.json()["results"]
        elevations.extend([r["elevation"] for r in results])
        time.sleep(1.1)  # don't hammer a free public instance

    # flat list -> back into a 2D grid matching (lats x lons) shape
    grid = np.array(elevations).reshape(resolution, resolution)
    return grid, lats, lons


if __name__ == "__main__":
    grid, lats, lons = fetch_elevation_grid(
        lat_min=31.10, lat_max=31.15,
        lon_min=77.15, lon_max=77.20,
        resolution=10
    )
    print("Elevation grid (meters):\n", grid)
    np.save("data/raw/lats.npy", lats)
    np.save("data/raw/lons.npy", lons)
    np.save("data/raw/elevation_grid.npy", grid)