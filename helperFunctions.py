import os
import pandas as pd


def extractMatchStats(league, date, hometeam, awayTeam):
    """
    Search for match information using league, date, hometeam and awayteam
    in the downloaded_data CSVs.
    """
    matchYear = pd.to_datetime(date).year # extract year to identify the seasons to search in
    seasonsToCheck = [ str(matchYear-2)[-2:] + str(matchYear-1)[-2:], # previous season
                      str(matchYear-1)[-2:] + str(matchYear)[-2:], # current season
                      str(matchYear)[-2:] + str(matchYear+1)[-2:] ] # next season

    # just to be safe, we will check for matching values in previous, current, next seasons.
    
    # we do this because there could be matches in June/July in the train_df 
    # We cannot definitively know what season they belong to.
    # Ideally, we should find only one row in the entire dataset of downloaded_data CSVs 
    # which should agree with the (league, date, hometeam, awayTeam) we are searching for.
    # Once we find it, we return it
    
    for season in seasonsToCheck:
        try:
            currSeasonCurrLeagueDF = pd.read_csv(os.path.join("downloaded_data", f"{league}_{season}.csv")).dropna(how='all')
        except FileNotFoundError: # for a match in 2009, we will check seasons 0708, 0809, 0910...
            continue # .. this function should not fail then.
        
        rowsDF = currSeasonCurrLeagueDF[    (currSeasonCurrLeagueDF["HomeTeam"] == hometeam) & \
                                            (currSeasonCurrLeagueDF["AwayTeam"] == awayTeam) & \
                                            (pd.to_datetime(currSeasonCurrLeagueDF["Date"]) == pd.to_datetime(date)) ]
        if len(rowsDF) > 0:
            return rowsDF


def getValuesForCurrentSeason(missingField, league, date, hometeam, awayTeam):
    """
    Search for match information using league, date, hometeam and awayteam
    in the downloaded_data CSVs. Return the values found in that column.
    """
    matchYear = pd.to_datetime(date).year # extract year to identify the seasons to search in
    seasonsToCheck = [ str(matchYear-2)[-2:] + str(matchYear-1)[-2:], # previous season
                      str(matchYear-1)[-2:] + str(matchYear)[-2:], # current season
                      str(matchYear)[-2:] + str(matchYear+1)[-2:] ] # next season
    
    for season in seasonsToCheck:
        try:
            currSeasonCurrLeagueDF = pd.read_csv(os.path.join("downloaded_data", f"{league}_{season}.csv")).dropna(how='all')
        except FileNotFoundError: # for a match in 2009, we will check seasons 0708, 0809, 0910...
            continue # .. this function should not fail then.
        
        # look for the match whose information we are trying to impute
        rowsDF = currSeasonCurrLeagueDF[    (currSeasonCurrLeagueDF["HomeTeam"] == hometeam) & \
                                            (currSeasonCurrLeagueDF["AwayTeam"] == awayTeam) & \
                                            (pd.to_datetime(currSeasonCurrLeagueDF["Date"]) == pd.to_datetime(date)) ]
        
        # once a row is found, we have found our season using which we need to calculate our stats
        if len(rowsDF) > 0:
            break
    
    teamLoc = "HomeTeam" if missingField[0] == "H" else "AwayTeam"
    currSeasonCurrLeagueCurrTeam = currSeasonCurrLeagueDF[currSeasonCurrLeagueDF[teamLoc] == (hometeam if teamLoc == "HomeTeam" else awayTeam)]
    
    return currSeasonCurrLeagueCurrTeam[missingField]
    