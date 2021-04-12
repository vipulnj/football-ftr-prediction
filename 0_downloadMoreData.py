# %% import necessary libraries

import os
import pandas as pd
# %% Build a list of league and season identifiers

# build a list of league identifiers as specified in www.football-data.co.uk data
leagueNames = ["bundesliga", "la-liga", "ligue-1", "premier-league", "serie-a"]
leagueIdentifierDict = { "bundesliga": "D1", "la-liga": "SP1", "ligue-1": "F1", "premier-league": "E0", "serie-a": "I1"}
leagueIdentifiers = [leagueIdentifierDict[leagueName] for leagueName in leagueNames] # list of league identifiers

# build a list of season identifiers i.e. 0809 for 2008-09 season, 0910 for 2009-10 season, 
seasonIdentifiers = [str(seasonStartYear)[-2:]+str(seasonStartYear+1)[-2:]   for seasonStartYear in range(2008, 2017+1)]

# %% download CSVs from football-data.co.uk and store the data. Notes about the same available in https://www.football-data.co.uk/notes.txt

if not os.path.exists('downloaded_data'):
    os.mkdir("downloaded_data") # create directory to save the CSVs

for leagueIx, leagueId in enumerate(leagueIdentifiers):
    for seasonIx, seasonId in enumerate(seasonIdentifiers):
        # the URL to access information is the following format
        link = f"https://www.football-data.co.uk/mmz4281/{seasonId}/{leagueId}.csv"
        saveLocation = os.path.join("downloaded_data", f"{leagueNames[leagueIx]}_{seasonId}.csv")
        pd.read_csv(link).to_csv(saveLocation, index=False) # read from the link and save it as CSV 

# %%
