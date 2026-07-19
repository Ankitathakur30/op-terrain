# src/geospatial/terrain_risk.py
import numpy as np

def compute_slope(elevation_grid,lats,lons):
    
    lat_spacing_m = abs(lats[1] - lats[0]) * 111_000  # ~111km per degree latitude, roughly constant
    lon_spacing_m = abs(lons[1] - lons[0]) * 111_000 * np.cos(np.radians(np.mean(lats)))  # shrinks with latitude

    dy, dx = np.gradient(elevation_grid, lat_spacing_m, lon_spacing_m)
    # combine both directions into a single slope magnitude, then to degrees
    slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_degrees = np.degrees(slope_radians)
    return slope_degrees


def classify_terrain_risk(slope_degrees):
    # explicit, auditable thresholds — not a learned model
    risk = np.empty(slope_degrees.shape, dtype=object)
    risk[slope_degrees < 5] = "low"
    risk[(slope_degrees >= 5) & (slope_degrees < 15)] = "moderate"
    risk[(slope_degrees >= 15) & (slope_degrees < 30)] = "high"
    risk[slope_degrees >= 30] = "severe"
    return risk


if __name__ == "__main__":
    grid = np.load("data/raw/elevation_grid.npy")
    lats = np.load("data/raw/lats.npy")
    lons = np.load("data/raw/lons.npy")
    slope = compute_slope(grid, lats, lons)
    risk = classify_terrain_risk(slope)
    print("Slope (degrees):\n", slope.round(1))
    print("\nRisk classification:\n", risk)