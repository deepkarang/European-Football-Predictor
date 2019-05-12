import sqlite3
import constants
import queries
import numpy as np
from sqlite3 import Error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from math import floor
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import isnan

#Initialize global constants
attributes = constants.attributes

#Initialize global variables
playerinfo = {}
label_vectorizer = {
    "W": [1,0,0],
    "D": [0,1,0],
    "L": [0,0,1]
}

def get_result_from_vector(vector):
    """
    Simple inverse map lookup function
    """
    for key in label_vectorizer:
        if label_vectorizer[key] == vector:
            return key


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
    :return: number of stats recorded per player
    """
    try:
        cur = conn.cursor()
        query = queries.fetch_players_performance
        cur.execute(query)
    except Error:
        print(Error)
        return
    rows = cur.fetchall()
    numstats = 0
    initialize_numstats = False
    for row in rows:
        playerstats = []
        for item in list(row):
            #Default value if DB does not have anything is 0
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
        if initialize_numstats is False:
            numstats = len(playerstats)
            initialize_numstats = True

    return numstats


def get_player_info(playerid):
    """
    look up player performance statistics in the playerinfo dict object
    :param playerid: numerical value of the player_api_id attribute
    :return: player average performance ratings
    """
    if playerid is None:
        return
    else:
        try:
            return playerinfo[playerid]
        except KeyError:
            return None


def get_match_data(conn,attrs):
    """
    Query the Match table data to get player analytics and results (features and labels)
    :param conn: the Connection object
    :param attrs: the attributes requested from the dataset
    :returns: player ratings, match result
    """
    cur = conn.cursor()
    query = "SELECT " + attrs + " FROM Match WHERE `id`>2000 LIMIT 10000"
    cur.execute(query)
    rows = cur.fetchall()
    matches = []
    results = []
    numwins = 0
    numdraws = 0
    numlosses = 0
    print("Pre-processing data...")
    for row in rows:
        row = list(row)
        goal_h = row[0]
        goal_a = row[1]
        match_outcome = [goal_h,goal_a]
        #Remove the 'goals scored' stats from the 'row' as they are labels not features
        del row[0]
        del row[0]
        #Retrieve match outcome and covert to binary representation for neural network
        result_text = get_match_result(match_outcome)
        #Used to keep track of how normalized the output classes in the dataset are
        if result_text == "W":
            numwins+=1
        else:
            if result_text == "D":
                numdraws+=1
            else:
                numlosses+=1
        res = label_vectorizer[result_text]
        players_performance = []
        #consolidate avg performance statistics for all players in the match
        for i in range(0,len(row)):
            stats = get_player_info(row[i])
            if stats is None:
                for i in range(0,numstats):
                    players_performance.append(0)
            else:   
                for stat in stats:
                    players_performance.append(stat)
        results.append(res)
        matches.append(players_performance)

    print("INPUT DATA RATIOS")
    print(numwins)
    print(numdraws)
    print(numlosses)

    return matches, results

        
def train_model(features, labels):
    """
    Train a Machine Learning model to return predictions for match score
    """
    print("Shape of input/output")
    print(features.shape)
    print(labels.shape)
    #Initialize model and define structure
    model = Sequential()
    model.add(Dense(32,input_dim=features.shape[1], kernel_initializer='normal'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.15))
    model.add(Dense(16,kernel_initializer='normal'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.15))
    model.add(Dense(16,kernel_initializer='normal'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(labels.shape[1], kernel_initializer='normal', activation='softmax'))
    optimizer = Adam(lr=0.000001)
    #Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #Print out summary of ANN structure
    model.summary()
    #Train model
    model.fit(features,labels,epochs=5,batch_size=256)
    return model

def test_model(model, features):
    """
    Test trained model against testing set
    """
    return model.predict(features)


def compare(predictions, expectations):
    """
    Evaluates how many predictions align with the corresponding expectations
    And prints out success rate
    """
    correct = 0
    total = 0
    for i in range(0,len(predictions)):
        total+=1
        outcome1 = expectations[i]
        outcome2 = predictions[i]
        j=0
        not_same = False
        for j in range(0,len(outcome1)):
            if outcome1[j] != outcome2[j]:
                not_same = True
        if not_same == False:
            correct+=1
    
    print("Correct: " + str(correct))
    print("Total: " + str(total))
    print("Accuracy : " + str(correct/total))

def get_match_result(match_result):
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

def cleanup_predictions(predictions):
    """
    Clean up output of neural network to determine most heavily weighted
    class and essentially the true 'prediction' for each input
    """
    i = 0
    for prediction in predictions:
        j=0
        cleaned_prediction = []
        desired_index = np.argmax(prediction)
        for val in prediction:
            if j == desired_index:
                cleaned_prediction.append(1)
            else:
                cleaned_prediction.append(0)
            j+=1
        predictions[i] = cleaned_prediction
        i+=1
    return predictions


def main():
    database = "database.sqlite"
    global numstats
    # create a database connection
    conn = create_connection(database)
    features, labels = [], []
    with conn:
        numstats = update_player_info(conn)
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
        cleaned_predictions = cleanup_predictions(predictions)
        compare(predictions, labels_test)

 
if __name__ == '__main__':
    main()

