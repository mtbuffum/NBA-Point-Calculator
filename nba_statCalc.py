from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import numpy as np
import matplotlib as plt
import copy,math



# Prompt user for custom stat input to predict PTS for the entire season
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

    # Assemble the input array in the same order as your training features
    new_stats = np.array([ast, reb, fg_pct, gp, plus_minus_rank, age, stl, blk, tov, fg_pct, fg3_pct, ft_pct])



except ValueError:
    print("Invalid input! Please enter numeric values for all stats.")


#Uses the API to opload all 500 NBA players stats (Information used to train model)
player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season="2024-25")
df = player_stats.get_data_frames()[0]
#Selects which categories to use
features = df[['PLAYER_NAME', 'PTS', 'AST', 'REB', 'FG_PCT', 'GP', 'PLUS_MINUS_RANK','AGE','STL','BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']]

#Append info to a CSV in order to see information and not exaughst the API limitaitons
csv_file = 'nba_player_stats_recent.csv'
features.to_csv(csv_file, index=False)

#Using Pandas read back from the data and create NUMPY arrays for training model
# Load the CSV
df = pd.read_csv('nba_player_stats_recent.csv')

# 1. Extract the 'PTS' column as a 1D NumPy array (target variable)
y_train = df['PTS'].to_numpy()

# 2. Drop non-stat columns and 'PTS' to get feature matrix
feature_cols = df.drop(columns=['PLAYER_NAME', 'PTS'])
X_train = feature_cols.to_numpy()



b_init = 785.1811367994083
w_init = np.zeros(X_train.shape[1])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")



#Cost Function J(w,b)
def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost

# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')


#Computes the gradient which is the derivatives of w,b based off of our cost function
def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw


#The actual gradient descent Function this applies the equation 
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

def main():
    # initialize parameters
    initial_w = np.zeros_like(w_init)
    initial_b = 0.
    # some gradient descent settings
    iterations = 1000
    alpha = 5.0e-7
    # run gradient descent 
    w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,compute_cost, compute_gradient, alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    m,_ = X_train.shape
    # Predict and print
    new_prediction = np.dot(new_stats, w_final) + b_final
    print(f"\nPredicted PTS for the season: {new_prediction:.2f}")



if __name__ == "__main__":
    main()



