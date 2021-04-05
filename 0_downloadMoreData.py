# %% import necessary libraries

import os
import pandas as pd

# %% Download information from football-data.co.uk 

# the url to access information from for different leagues and seasons has the structure https://www.football-data.co.uk/mmz4281/{season-identifier}/{league-identifier}.csv

# build a list of league identifiers
leagueNames = ["bundesliga", "la-liga", "ligue-1", "premier-league", "serie-a"]
leagueIdentifierDict = { "bundesliga": "D1", "la-liga": "SP1", "ligue-1": "F1", "premier-league": "E0", "serie-a": "I1"}
leagueIdentifiers = [leagueIdentifierDict[leagueName] for leagueName in leagueNames] # list of league identifiers

# build a list of season identifiers
seasonIdentifiers = [str(seasonStartYear)[-2:]+str(seasonStartYear+1)[-2:]   for seasonStartYear in range(2008, 2017+1)]

# %% download CSVs from football-data.co.uk and store the data. Notes about the same available in https://www.football-data.co.uk/notes.txt

os.mkdir("downloaded_data") # create directory to save the CSVs

for leagueIx, leagueId in enumerate(leagueIdentifiers):
    for seasonIx, seasonId in enumerate(seasonIdentifiers):
        link = f"https://www.football-data.co.uk/mmz4281/{seasonId}/{leagueId}.csv"
        saveLocation = os.path.join("downloaded_data", f"{leagueNames[leagueIx]}_{seasonId}.csv")
        pd.read_csv(link).to_csv(saveLocation, index=False)

# %%
