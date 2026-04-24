import math

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import socceraction.spadl.config as spadlcfg
from mplsoccer import Pitch


import matplotlib.pyplot as plt

# --- Config ---
DATA_PATH = Path().resolve() / "data" / "vaep_data" / "all_leagues_vaep.h5"
ACTION_TYPES = ["pass", "cross"]
SEASON_NAME = "2015/2016"
COMPETITIONS = ["Serie A", "La Liga", "Premier League", "1. Bundesliga", "Ligue 1"]
N_CLUSTERS = 50  # adjustable
TOP_K = 5

# --- Load data ---
with pd.HDFStore(DATA_PATH, mode="r") as store:
    df = store['vaep_data']

actiontype_map = dict(enumerate(spadlcfg.actiontypes))
df["action_type"] = df["type_id"].map(actiontype_map).fillna("unknown")

# --- Filter data ---
df = df[
    (df['action_type'].isin(ACTION_TYPES)) &
    (df['season_name'] == SEASON_NAME) &
    (df['competition_name'].isin(COMPETITIONS))
].copy()

# --- Feature engineering ---
# Option 1: start_x, start_y, dx, dy
df['dx'] = df['end_x_a0'] - df['start_x_a0']
df['dy'] = df['end_y_a0'] - df['start_y_a0']

# Option 2: start_x, start_y, length, cos, sin
df['length'] = np.sqrt(df['dx']**2 + df['dy']**2)
df['cos'] = df['dx'] / (df['length'] + 1e-8)
df['sin'] = df['dy'] / (df['length'] + 1e-8)

# Choose features for clustering
FEATURES = ['start_x_a0', 'start_y_a0', 'dx', 'dy']
# FEATURES = ['start_x_a0', 'start_y_a0', 'length', 'cos', 'sin']

def run_clustering(df, n_clusters=50, seed=123):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[FEATURES])
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    df = df.copy()
    df["cluster"] = model.fit_predict(X)
    return df, scaler


def compute_zscores(df):
    team_cluster = (
        df.groupby(["competition_name", "cluster"])
        .size()
        .reset_index(name="count")
    )
    team_total = df.groupby("competition_name").size().reset_index(name="total")
    dist = team_cluster.merge(team_total, on="competition_name")
    dist["prop"] = dist["count"] / dist["total"]
    league_avg = dist.groupby("cluster")["prop"].mean()
    league_std = dist.groupby("cluster")["prop"].std()
    dist["z"] = (
        (dist["prop"] - dist["cluster"].map(league_avg)) /
        (dist["cluster"].map(league_std) + 1e-6)
    )
    return dist

def representative_passes(df_cluster, scaler, n=30):
    X = scaler.transform(df_cluster[FEATURES])
    centroid = X.mean(axis=0)
    dist = np.linalg.norm(X - centroid, axis=1)
    df_cluster = df_cluster.copy()
    df_cluster["dist"] = dist
    return df_cluster.sort_values("dist").head(n)


pitch = Pitch(pitch_type="statsbomb", line_color="black")

def plot_team(ax, df, dist, scaler, league, display_name=None,
              top_n=5, show_descriptions=False):
    team_df = df[df["competition_name"] == league].copy()       
    # --- handle z column naming ---
    z_col = "z_score" if "z_score" in dist.columns else "z"
    top_clusters = (
        dist[dist["competition_name"] == league]
        .sort_values(z_col, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    cmap = plt.colormaps["tab10"]
    # --- stable cluster-color mapping ---
    cluster_info = []
    for i, row in enumerate(top_clusters.itertuples()):
        cluster_info.append({
            "cluster": row.cluster,
            "color": cmap(i % 10),
            "idx": i + 1
        })
    # --- plot in reverse so #1 is on top ---
    for info in reversed(cluster_info):
        cluster_df = team_df[team_df["cluster"] == info["cluster"]]
        reps = representative_passes(cluster_df, scaler)
        for _, p in reps.iterrows():
            ax.arrow(
                p["start_x_a0"], p["start_y_a0"],
                p["dx"], p["dy"],
                head_width=1.2,
                head_length=2,
                linewidth=1.3,
                color=info["color"],
                alpha=0.4
            )
    # --- legend (original order) ---
    descriptions = [(c["idx"], c["color"]) for c in cluster_info]
    # --- title (with optional position) ---
    title = display_name if display_name is not None else league    
    ax.set_title(title, pad=6, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    y = -0.02
    if show_descriptions:
        for i, (idx, color) in enumerate(descriptions):
            ax.text(
                0.5,
                y - i * 0.06,
                f"➜ {idx}",
                transform=ax.transAxes,
                fontsize=9,
                color=color,
                ha="center",
                va="top"
            )
    else:
        x_positions = np.linspace(0.2, 0.8, len(descriptions))
        for x, (idx, color) in zip(x_positions, descriptions):
            ax.text(
                x,
                y,
                f"➜ {idx}",
                transform=ax.transAxes,
                fontsize=11,
                color=color,
                ha="center",
                va="top"
            )



def plot_grid(df, dist, scaler, title, n_cols=4, show_descriptions=False):
    TEAM_ORDER = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
    teams = [t for t in TEAM_ORDER if t in df["competition_name"].unique()]
    n_rows = math.ceil(len(teams) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 4 * n_rows),
        gridspec_kw={'hspace': 0.08, 'wspace': 0.05}
    )
    axes = axes.flatten()
    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=0.91
    )
    for i, team in enumerate(teams):
        pitch.draw(ax=axes[i])
        plot_team(
            axes[i],
            df,
            dist,
            scaler,
            team,
            display_name=f"{i+1}. {team}",
            show_descriptions=show_descriptions
        )
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.show()


df_all, scaler = run_clustering(df)


dist_all = compute_zscores(df_all)

plot_grid(
    df_all,
    dist_all,
    scaler,
    "Team Passing Style Clusters (All Passes) | La Liga 2015/16 | "
    "Top 5 Clusters (Z-Score vs League)"
)

# # --- Clustering and analysis ---
# for league in COMPETITIONS: 
#     league_df = df[df['competition_name'] == league]
#     X = league_df[FEATURES].values
#     # Clustering
#     kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
#     league_df['cluster'] = kmeans.fit_predict(X)
#     cluster_centers = kmeans.cluster_centers_
#     # League centroid
#     league_centroid = X.mean(axis=0)
#     # Distance from each cluster centroid to league centroid
#     dists = np.linalg.norm(cluster_centers - league_centroid, axis=1)
#     topk_idx = np.argsort(dists)[-TOP_K:]
#     print(f"\n{league}:")
#     print("Top 5 clusters (most distant from league centroid):", topk_idx)
#     # Plot
#     plt.figure(figsize=(8, 6))
#     for idx in topk_idx:
#         cluster_actions = league_df[league_df['cluster'] == idx].sample(n=100, random_state=42)  # sample for visualization
#         for _, row in cluster_actions.iterrows():
#             _ = plt.arrow(
#                 row['start_x_a0'], row['start_y_a0'],
#                 row['dx'], row['dy'],
#                 head_width=1.5, head_length=2, alpha=0.2, color=f"C{idx}"
#             )
#     plt.title(f"{league} - Top {TOP_K} Outlier Clusters (start-end)")
#     plt.xlabel("start_x_a0")
#     plt.ylabel("start_y_a0 ")
#     plt.xlim(0, 105)
#     plt.ylim(0, 68)
#     plt.gca().set_aspect('equal')
#     plt.show()
#     break
