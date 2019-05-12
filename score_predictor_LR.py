import sqlite3
import constants
import queries
import numpy as np
from sqlite3 import Error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import floor
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import isnan

#Initialize global constants
attributes = constants.attributes

#Initialize global variables
playerinfo = {}

def is_valid_attribute(col):
    """
    Check if the given column is a desired attribute
    :param col: column name
    :return: boolean indicating validity
    """
    result = False
    for attribute in attributes:
        if col == attribute:
            result = True
            break
    
    return result

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None
 
 
def get_match_attributes(conn):
    """
    Query Match table columns
    :param conn: the Connection object
    :return: string with all attributes concatonated
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(Match)")
    rows = cur.fetchall()
    result = ""
    columns = []
    for row in rows:
        columns.append(row[1])
        if is_valid_attribute(row[1]):
            result+=row[1]
            result+=", "

    result = result[0:len(result)-2]
    return result

def update_player_info(conn):
    """
    update the playerinfo dict object with keys as the player_api_id
    and values as the corresponding performance stats for that player
    :param conn: the Connection object
    """
    try:
        cur = conn.cursor()
        query = "SELECT player_api_id, AVG(overall_rating), AVG(crossing), AVG(reactions), AVG(stamina), AVG(strength), AVG(finishing), AVG(heading_accuracy), AVG(long_passing), AVG(short_passing), AVG(dribbling), AVG(ball_control), AVG(agility), AVG(interceptions), AVG(sliding_tackle) FROM Player_Attributes GROUP BY player_api_id"
        cur.execute(query)
    except Error:
        print(Error)
        return
    rows = cur.fetchall()
    for row in rows:
        playerstats = []
        for item in list(row):
            if item is None:
                playerstats.append(0)
                continue
            if isnan(item):
                playerstats.append(0)
            else:
                playerstats.append(item)
        #remove player_api_id from stats
        del playerstats[0]
        playerinfo[row[0]] = playerstats


def get_player_info(playerid):
    """
    look up player performance statistics in the playerinfo dict object
    :param playerid: numerical value of the player_api_id attribute
    """
    if playerid is None:
        return
    else:
        return playerinfo[playerid]


def get_match_data(conn,attrs):
    """
    Query the Match table data to get player analytics and results (features and labels)
    :param conn: the Connection object
    :param attrs: the attributes requested from the dataset
    :returns: player ratings, match result
    """
    cur = conn.cursor()
    query = "SELECT " + attrs + " FROM Match WHERE `id`>2000 LIMIT 2000"
    cur.execute(query)
    rows = cur.fetchall()
    matches = []
    results = []
    print("Pre-processing data...")
    for row in rows:
        row = list(row)
        goal_h = row[0]
        goal_a = row[1]
        #Remove the 'goals scored' stats from the 'row' as they are labels not features
        del row[0]
        del row[0]
        res = [goal_h,goal_a]
        players_performance = []
        #get avg match performance statistics for all players 
        for i in range(0,len(row)):
            stats = get_player_info(row[i])
            if stats is None:
                for i in range(0,14):
                    players_performance.append(0)
            else:       
                for stat in stats:
                    players_performance.append(stat)
        results.append(res)
        matches.append(players_performance)

    return matches, results

        
def train_model(features, labels):
    """
    Train a Machine Learning model to return predictions for match score
    """
    model = LinearRegression().fit(features,labels)
    return model

def test_model(model, features):
    """
    Test trained model against testing set
    """
    predictions = model.predict(features)
    return predictions


def compare(predictions, expectations):
    """
    Evaluates how many predictions align with the corresponding expectations
    And prints out success rate
    """
    correct = 0
    total = 0
    for i in range(0,len(predictions)):
        total+=1
        outcome1 = get_match_outcome(expectations[i])
        outcome2 = get_match_outcome(predictions[i])
        if outcome1 == outcome2:
            correct+=1
    
    print("Correct: " + str(correct))
    print("Total: " + str(total))
    print("Accuracy : " + str(correct/total))

def get_match_outcome(match_result):
    """
    Returns whether the home team won, lost, or drew the match in question
    The first index in match_result is the score of the home team and second 
    Index is the score of the away team
    :param match_result: 2-item array containing points scored by home team and away team
    :returns: string "W", "D", or "L"
    """
    if match_result[0] > match_result[1]:
        return "W"
    else:
        if match_result[0] == match_result[1]:
            return "D"
        else: 
            return "L"


def main():
    database = "database.sqlite"
 
    # create a database connection
    conn = create_connection(database)
    features, labels = [], []
    with conn:
        update_player_info(conn)
        attrs = get_match_attributes(conn)

        print("Fetching data for all Matches...")
        features, labels = get_match_data(conn,attrs)

        print("Splitting into training and testing sets...")
        split = floor(0.8*len(features))
        features_train = np.array(features[0:split])
        labels_train = np.array(labels[0:split])
        features_test = np.array(features[split:len(features)])
        labels_test = np.array(labels[split:len(labels)])

        print("Training and Testing Regression model...")
        model = train_model(features_train,labels_train)
        predictions = test_model(model, features_test)

        print("Mean squared error: " +  str(mean_squared_error(predictions, labels_test)))
        print('Variance score:' + str(r2_score(labels_test, predictions)))
        compare(predictions, labels_test)

 
if __name__ == '__main__':
    main()

