# DCF 군집화 + 거점 산출
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform
from scipy.spatial import ConvexHull, QhullError
from sklearn.metrics.pairwise import haversine_distances
from config import (
    DELIVERY_TARGET_MINUTES, SPEED_FACTOR, COMPLEXITY_FACTOR, DETOUR_FACTOR,
)


def calc_max_distance_km():
    dcf = (COMPLEXITY_FACTOR * DETOUR_FACTOR) / SPEED_FACTOR
    return (DELIVERY_TARGET_MINUTES / 60) / dcf


def run_clustering(df):
    df = df.copy()
    df['lat_rad'] = np.radians(df['lat'])
    df['lon_rad'] = np.radians(df['lon'])

    coords_rad  = df[['lat_rad', 'lon_rad']].values
    dist_matrix = haversine_distances(coords_rad) * 6371.0088
    dist_array  = squareform(dist_matrix)
    Z           = linkage(dist_array, method='complete')
    max_dist_km = calc_max_distance_km()

    df['cluster'] = fcluster(Z, t=max_dist_km, criterion='distance')
    return df, Z, max_dist_km


def build_spots(df):
    total_stores = len(df)
    spots = []

    for cid in sorted(df['cluster'].unique()):
        cdf    = df[df['cluster'] == cid]
        points = cdf[['lat', 'lon']].values

        spots.append({
            "cluster_id":  int(cid),
            "lat":         float(cdf['lat'].mean()),
            "lon":         float(cdf['lon'].mean()),
            "count":       len(cdf),
            "hull":        _calc_hull(points),
            "store_ratio": round(len(cdf) / total_stores * 100, 1),
        })
    return spots


def evaluate_clustering(df, Z, max_dist_km):
    df = df.copy()
    df['lat_rad'] = np.radians(df['lat'])
    df['lon_rad'] = np.radians(df['lon'])

    dist_array = squareform(
        haversine_distances(df[['lat_rad', 'lon_rad']].values) * 6371.0088
    )
    cpcc, _ = cophenet(Z, dist_array)

    violation_count, max_observed = 0, 0.0
    for cid in df['cluster'].unique():
        cdf = df[df['cluster'] == cid]
        if len(cdf) < 2:
            continue
        pair_dists  = haversine_distances(cdf[['lat_rad', 'lon_rad']].values) * 6371.0088
        cluster_max = pair_dists.max()
        max_observed = max(max_observed, cluster_max)
        if cluster_max > max_dist_km:
            violation_count += 1

    return {
        "cpcc":            round(cpcc, 4),
        "max_dist_km":     round(max_dist_km, 4),
        "max_observed_km": round(max_observed, 4),
        "violation_count": violation_count,
    }


def _calc_hull(points):
    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
            return [{"lat": float(points[v][0]), "lon": float(points[v][1])} for v in hull.vertices]
        except QhullError:
            pass
    return [{"lat": float(p[0]), "lon": float(p[1])} for p in points]