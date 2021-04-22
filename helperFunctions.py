import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score


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
    if not math.isnan(homeWinOdds) and not math.isnan(drawOdds) and not math.isnan(awayWinOdds):
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


def maxDstDrw(row):
    """
    Returns max distance-from-drawn-match value based on betting odds captured by Bet365.
    Positive values mean homeTeam has that much more than chances of winning instead of a draw.
    Negative values means awayTeam has that much more than chances of winning instead of a draw. Negative value here does not less chances (the absolute value must be taken) but signals an awayTeam winning chances over a draw.
    """
    B365HDstDrw, B365ADstDrw = row['B365HDstDrw'], row['B365ADstDrw']
    dstDrw = np.max([B365HDstDrw, B365ADstDrw])
    dstDrwIx = np.argmax([B365HDstDrw, B365ADstDrw])
    signDict = {0: 1, 1: -1}
    return signDict[dstDrwIx]*dstDrw


def normalizeConfusionMatrix(cm):
    ncm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ncm = np.around(ncm, decimals=3)
    ncm[np.isnan(ncm)] = 0
    return ncm


def calculate_accuracy(confmat): 
    return f"Accuracy = {(round(np.trace(confmat)/np.sum(confmat), 3))*100}%"


def plotROCcurve(yTrue_scores, yPred_scores, classesInOrder, classNumToName, title=None):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    title = "ROC curve" if title is None else title
    
    if pd.api.types.is_object_dtype(classesInOrder):
        classDict = {'A': -1, 'D': 0, 'H': 1} 
        classNumToName = {classDict[classStr]:classNumToName[classStr] for classStr in classesInOrder}
        classesInOrder = [classDict[classStr] for classStr in classesInOrder]
    
    fpr, tpr, roc_auc = {}, {}, {}
    for classLabel in classesInOrder:
        fpr[classLabel], tpr[classLabel], _ = roc_curve(yTrue_scores[:, classLabel+1], yPred_scores[:, classLabel+1])
        roc_auc[classLabel] = auc(fpr[classLabel], tpr[classLabel])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(yTrue_scores.ravel(), yPred_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot all ROC curves
    plt.figure(figsize=(6, 6), dpi=100)
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='gold', linestyle=':', linewidth=4)
    
    # plot the ROC curves for each class with colors specified
    for classLabel, color in zip(classesInOrder, ['red', 'green', 'blue']): # these are colors represeting AwayWin, Draw, HomeWin
        plt.plot(fpr[classLabel], tpr[classLabel], color=color, label=f"{classNumToName[classLabel]} AUC = {str(np.round(roc_auc[classLabel], 3))[:4]}")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate ')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plotPRcurve(yTrue_scores, yPred_scores, classesInOrder, classNumToName, title=None):
    """
    Taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """
    
    title = "PR curve" if title is None else title
    
    
    if pd.api.types.is_object_dtype(classesInOrder):
        classDict = {'A': -1, 'D': 0, 'H': 1} 
        classNumToName = {classDict[classStr]:classNumToName[classStr] for classStr in classesInOrder}
        classesInOrder = [classDict[classStr] for classStr in classesInOrder]
    
    precision, recall, average_precision = {}, {}, {}
    
    for classLabel in classesInOrder:
        precision[classLabel], recall[classLabel], _ = precision_recall_curve(yTrue_scores[:, classLabel+1], yPred_scores[:, classLabel+1])
        average_precision[classLabel] = average_precision_score(yTrue_scores[:, classLabel+1], yPred_scores[:, classLabel+1], average="micro")
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(yTrue_scores.ravel(), yPred_scores.ravel())
    average_precision["micro"] = average_precision_score(yTrue_scores, yPred_scores, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
    
    plt.figure(figsize=(6, 10), dpi=100)
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', linewidth=4, linestyle=':', )
    """plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='gold', linestyle=':', linewidth=4)"""
    lines.append(l)
    labels.append('Overall AUC i.e. micro AP for all classes = {0:0.2f})'.format(average_precision["micro"]))
    
    for classLabel, color in zip(classesInOrder, ['red', 'green', 'blue']):
        l, = plt.plot(recall[classLabel], precision[classLabel], color=color, lw=2)
        lines.append(l)
        labels.append('{0} (AUC i.e. micro AP for = {1:0.2f})'.format(classNumToName[classLabel], average_precision[classLabel]))
    
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))