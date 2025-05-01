from nba_api.stats.endpoints import leaguedashplayerstats
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Prompt user for input
print("\nEnter the following stats to predict a player's points for the season:")
try:
    ast = float(input("Assists for the season (AST): "))
    reb = float(input("Rebounds for the season (REB): "))
    fg_pct = float(input("Field Goal % AS A DECIMAL(FG_PCT): "))
    gp = float(input("Games Played (GP): "))
    plus_minus_rank = float(input("PLUS_MINUS_RANK: "))
    age = float(input("Age: "))
    stl = float(input("Steals for the season (STL): "))
    blk = float(input("Blocks for the season (BLK): "))
    tov = float(input("Turnovers for the season (TOV): "))
    fg3_pct = float(input("3PT % AS A DECIMAL (FG3_PCT): "))
    ft_pct = float(input("Free Throw % AS A DECIMAL(FT_PCT): "))

    # Input array for prediction (order must match feature training columns)
    new_stats = np.array([[ast, reb, fg_pct, gp, plus_minus_rank, age, stl, blk, tov, fg_pct, fg3_pct, ft_pct]])
except ValueError:
    print("Invalid input! Please enter numeric values for all stats.")
    exit()

# Get player stats from NBA API
player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season="2024-25")
df = player_stats.get_data_frames()[0]

# Select relevant features
features = df[['PLAYER_NAME', 'PTS', 'AST', 'REB', 'FG_PCT', 'GP', 'PLUS_MINUS_RANK',
               'AGE', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']]
features.to_csv('nba_player_stats_recent.csv', index=False)  # Cache for reuse

# Load from CSV
df = pd.read_csv('nba_player_stats_recent.csv')

# Target variable and feature matrix
y_train = df['PTS'].to_numpy()
X_train = df.drop(columns=['PLAYER_NAME', 'PTS']).to_numpy()

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the model
prediction = model.predict(new_stats)
print(f"\nPredicted PTS for the season: {prediction[0]:.2f}")
