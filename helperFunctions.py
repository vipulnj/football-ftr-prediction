import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import isnan


def searchMatch(league, date, homeTeam, awayTeam):
    """ 
    Search for match information based on league, date, homeTeam, awayTeam.
    This works because there is only one match in a season where there is homeTeam vs. awayTeam.
    """ 
    matchYear = pd.to_datetime(date).year # extract year to identify the seasons to search in 
    seasonStartedPrevYear = str(matchYear-1)[-2:] + str(matchYear)[-2:] # started last August
    seasonStartedThisYear = str(matchYear)[-2:] + str(matchYear+1)[-2:] # starting/started this August
    
    seasonToCheck = seasonStartedPrevYear if pd.to_datetime(date).month <= 6 else seasonStartedThisYear
    currSeasonDF = pd.read_csv(os.path.join("downloaded_data", f"{league}_{seasonToCheck}.csv")).dropna(how='all')
    try:
        currSeasonDF['Date'] = pd.to_datetime(currSeasonDF['Date'], format="%d/%m/%y") # change object to datetime dtype
    except ValueError as err:
        currSeasonDF['Date'] = pd.to_datetime(currSeasonDF['Date'], format="%d/%m/%Y") # 4 chars to represent year is some CSVs

    # search for the match for each season's CSV
    matchInfoDF = currSeasonDF[ 
                    (currSeasonDF["HomeTeam"] == homeTeam) & (currSeasonDF["AwayTeam"] == awayTeam) & \
                    (currSeasonDF["Date"] == pd.to_datetime(date))    ].reset_index()
    
    if len(matchInfoDF) != 1:
        print(f"{league} match between {homeTeam} and {awayTeam} on {pd.to_datetime(date).strftime('%d %B %Y')} in \
            {seasonToCheck} CSV returned more than {len(matchInfoDF)} results")
        if len(matchInfoDF) == 0:
            return None, None
        elif len(matchInfoDF) > 1:
            raise Exception("More than one row for the match! Possible issue with the data.")
    
    # exactly one row was found
    return matchInfoDF, seasonToCheck


def extractMatchOdds(row):
    """
    Extract Bet365 matchOdds information after searching for the match. 
    We use Bet365 because that is the one available for all matches across leagues.
    """
    matchInfoDF, _ = searchMatch(row["league"], row["Date"], row["HomeTeam"], row["AwayTeam"])
    if matchInfoDF is None:
        return None, None, None
    return matchInfoDF.iloc[0]["B365H"], matchInfoDF.iloc[0]["B365D"], matchInfoDF.iloc[0]["B365A"]


def extractMissingMatchStatsForCurrentMatch(missingFields, row):
    """
    Extract missing value for the specified field after searching for the match. 
    We do this to just "get" the value the training data might be missing and is available in downloaded_data
    """
    league, date, homeTeam, awayTeam = row["league"], row["Date"], row["HomeTeam"], row["AwayTeam"]
    matchInfoDF, _ = searchMatch(league, date, homeTeam, awayTeam)
    return matchInfoDF[missingFields] # get missingFields' values and return them
    
    
def accessValuesUnderThisFieldForThisSeason(field, row):
    """
    Access all values under the specified field for the team that season. These can be used impute the missing values.
    """
    league, date, homeTeam, awayTeam = row["league"], row["Date"], row["HomeTeam"], row["AwayTeam"]
    _, season = searchMatch(league, date, homeTeam, awayTeam)
    currSeasonDF = pd.read_csv(os.path.join("downloaded_data", f"{league}_{season}.csv")).dropna(how='all')
    homeOrAway = "HomeTeam" if field[0] == "H" else "AwayTeam" # should we search stats for HomeTeam or AwayTeam?
    teamToSearch = homeTeam if homeOrAway == "HomeTeam" else awayTeam # set the var depending on Home or Away
    return currSeasonDF[currSeasonDF[homeOrAway] == teamToSearch][field] # access all home matches' or all away matches' stats for the team


def numWDLInLastFiveMatches(row):
    """
    Look back the last five matches for both homeTeam and awayTeam and count number of wins, losses and draws.
    """
    league, date, homeTeam, awayTeam = row["league"], row["Date"], row["HomeTeam"], row["AwayTeam"]
    _, season = searchMatch(league, date, homeTeam, awayTeam) # we get the season we found this match in
    currSeasonDF = pd.read_csv(os.path.join("downloaded_data", f"{league}_{season}.csv")).dropna(how='all')
    try:
        currSeasonDF['Date'] = pd.to_datetime(currSeasonDF['Date'], format="%d/%m/%y") # change object to datetime dtype
    except ValueError as err:
        currSeasonDF['Date'] = pd.to_datetime(currSeasonDF['Date'], format="%d/%m/%Y") # 4 chars to represent year is some CSVs
    
    numWDLs = [] # first element for HomeTeam, second element for AwayTeam
    for currTeam in [homeTeam, awayTeam]:
        numWins, numDraws, numLosses = 0, 0, 0     # for current team
        prevFiveMatchesDF = currSeasonDF[
            # look for matches of the currTeam irrespective of HomeMatch or AwayMatch ... 
            ((currSeasonDF["HomeTeam"] == currTeam) | (currSeasonDF["AwayTeam"] == currTeam)) \
            # ... look for dates smaller than current date to get matches played by currTeam before today's match
            & (currSeasonDF["Date"] < pd.to_datetime(date))
        ].sort_values(by='Date', ascending=True).tail(n=5) # look back only last five matches
        
        for ix, row in prevFiveMatchesDF.iterrows():
            # if the match is a draw, record a draw for the currTeam
            numDraws += 1 if prevFiveMatchesDF.at[ix, "FTR"] == "D" else 0
            # if currTeam is homeTeam (or awayTeam) and FTR is H (or A), then record a win for the currTeam
            numWins += 1 if currTeam == prevFiveMatchesDF.at[ix, "HomeTeam"] and prevFiveMatchesDF.at[ix, "FTR"] == "H" else 0
            numWins += 1 if currTeam == prevFiveMatchesDF.at[ix, "AwayTeam"] and prevFiveMatchesDF.at[ix, "FTR"] == "A" else 0
            # if currTeam is homeTeam (or awayTeam) and FTR is A (or H), then record a loss for the currTeam
            numLosses += 1 if prevFiveMatchesDF.at[ix, "FTR"] == "A" and prevFiveMatchesDF.at[ix, "HomeTeam"] == currTeam else 0
            numLosses += 1 if prevFiveMatchesDF.at[ix, "FTR"] == "H" and prevFiveMatchesDF.at[ix, "AwayTeam"] == currTeam else 0
        # record wins, draws and losses for the currTeam
        numWDLs.append([numWins, numDraws, numLosses])
    return numWDLs[0] + numWDLs[1]


def numDaysSinceLastMatch(row):
    """
    Get the number of days since the homeTeam played a match. Similarly, get it for the awayTeam.
    """
    league, date, homeTeam, awayTeam = row["league"], row["Date"], row["HomeTeam"], row["AwayTeam"]
    _, season = searchMatch(league, date, homeTeam, awayTeam) # we get the season we found this match in
    currSeasonDF = pd.read_csv(os.path.join("downloaded_data", f"{league}_{season}.csv")).dropna(how='all')
    try:
        currSeasonDF['Date'] = pd.to_datetime(currSeasonDF['Date'], format="%d/%m/%y") # change object to datetime dtype
    except ValueError as err:
        currSeasonDF['Date'] = pd.to_datetime(currSeasonDF['Date'], format="%d/%m/%Y") # 4 chars to represent year is some CSVs
    
    numDaysSinceLastMatchCounts = [] # first element for HomeTeam, second element for AwayTeam
    for currTeam in [homeTeam, awayTeam]:
        prevMatchDF = currSeasonDF[
            # look for matches of the currTeam irrespective of HomeMatch or AwayMatch ... 
            ((currSeasonDF["HomeTeam"] == currTeam) | (currSeasonDF["AwayTeam"] == currTeam)) \
            # ... look for dates smaller than current date to get the match played by currTeam before today's match
            & (currSeasonDF["Date"] < pd.to_datetime(date))
        ].sort_values(by='Date', ascending=True).tail(n=1) # look back only one match
        
        if len(prevMatchDF) == 0: # probably the start of the season
            numDaysSinceLastMatchCounts.append(18) # 2 to 3 weeks between end of club friendles/world cup and start the leagues
        elif len(prevMatchDF) == 1:
            # print(">>", pd.to_datetime(prevMatchDF["Date"].values[0]).strftime("%d %B %Y"))
            numDaysSinceLastMatch_currTeam = (pd.to_datetime(date) - pd.to_datetime(prevMatchDF["Date"].values[0])).days
            # print(f"numDaysSinceLastMatch_currTeam = {numDaysSinceLastMatch_currTeam}")
            numDaysSinceLastMatchCounts.append(numDaysSinceLastMatch_currTeam)
        elif len(prevMatchDF) > 1:
            raise Exception("More than one rows should not be found!")
    return numDaysSinceLastMatchCounts


def getBet365prediction(row):
    """
    Return most predictable result according to Bet365.
    """
    homeWinBetOdds, drawBetOdds, awayWinBetOdds = row["B365H"], row["B365D"], row["B365A"]
    odds = [homeWinBetOdds, drawBetOdds, awayWinBetOdds]
    results = ['H', 'D', 'A']
    # least bettingOdds means that result is most predictable
    bet365prediction = results[odds.index(min(odds))]
    return bet365prediction


def extractAverageBettingOddsIfMissing(row):
    """
    Return the average betting odds (over last 3 years) when these two teams meet at the HomeTeam's stadium.
    """
    homeWinOdds, drawOdds, awayWinOdds = row["B365H"], row["B365D"], row["B365A"]
    # since we know only these kind of rows are present in train_df, we can specify this condition
    if not isnan(homeWinOdds) and not isnan(drawOdds) and not isnan(awayWinOdds):
        return homeWinOdds, drawOdds, awayWinOdds # just return what you read
    
    league, date, homeTeam, awayTeam = row["league"], row["Date"], row["HomeTeam"], row["AwayTeam"]
    _, season = searchMatch(league, date, homeTeam, awayTeam) # we get the season we found this match in
    seasonStart, seasonEnd = season[:2], season[2:]
    
    seasonsToCheck = [
        str(int(seasonStart)-1).zfill(2) + str(int(seasonEnd)-1).zfill(2), # previous season
        str(int(seasonStart)-2).zfill(2) + str(int(seasonEnd)-2).zfill(2), # 2 seasons ago
        str(int(seasonStart)-3).zfill(2) + str(int(seasonEnd)-3).zfill(2), # 3 seasons ago
    ]
    
    homeWinOddsPrevSeasons, drawOddsPrevSeasons, awayWinOddsPrevSeasons = [], [], []
    for season in seasonsToCheck:
        try:
            prevSeasonDF = pd.read_csv(os.path.join("downloaded_data", f"{league}_{season}.csv")).dropna(how='all')
            sameMatchPrevSeasonDF = prevSeasonDF[(prevSeasonDF["HomeTeam"] == homeTeam) & (prevSeasonDF["AwayTeam"] == awayTeam)].reset_index()
            if len(sameMatchPrevSeasonDF) == 0:
                continue # probably that match did not happen since one or both team were not in the league i.e. got relegated
            if len(sameMatchPrevSeasonDF) > 1:
                raise Exception("More than one match found!!")
            
            homeWinOdds = sameMatchPrevSeasonDF["B365H"].values[0]
            drawOdds = sameMatchPrevSeasonDF["B365D"].values[0]
            awayWinOdds = sameMatchPrevSeasonDF["B365A"].values[0]
            
            if not (homeWinOdds is None and drawOdds is None and awayWinOdds is None):
                homeWinOddsPrevSeasons.append(homeWinOdds)
                drawOddsPrevSeasons.append(drawOdds)
                awayWinOddsPrevSeasons.append(awayWinOdds)
        except FileNotFoundError as err:
            break # no use searching an older season
    
    avgHomeOdds, avgDrawOdds, avgAwayOdds = 0.0, 0.0, 0.0
    avgHomeOdds = np.average(homeWinOddsPrevSeasons) if len(homeWinOddsPrevSeasons) >= 1 else avgHomeOdds
    avgDrawOdds = np.average(drawOddsPrevSeasons) if len(drawOddsPrevSeasons) >= 1 else avgDrawOdds
    avgAwayOdds = np.average(awayWinOddsPrevSeasons)  if len(awayWinOddsPrevSeasons) >= 1 else avgAwayOdds
    return avgHomeOdds, avgDrawOdds, avgAwayOdds


def normalizeConfusionMatrix(cm):
    ncm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ncm = np.around(ncm, decimals=3)
    ncm[np.isnan(ncm)] = 0
    return ncm


def calculate_accuracy(confmat): 
    return f"Accuracy = {(round(np.trace(confmat)/np.sum(confmat), 3))*100}%"