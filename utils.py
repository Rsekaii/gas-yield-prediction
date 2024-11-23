import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from model import FeedForwardNN


### Pre-processing Functions ###

def rm_zeros(df, columns):
    return df[(df[columns] > 0).all(axis=1)].copy()

def rm_outliers(df, n_neighbors=20, contamination=0.01):
    factor = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    filter = factor.fit_predict(df.select_dtypes(include=[np.number]))

    return df[filter != -1]

def uni_normalization(df, numerical_cols, epsilon=1e-6):
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols]) + epsilon
    return


### Neural Network Score Functions ###

def accuracy_nn(y_pred, y_true):
    return 100 - torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

def r2_score_nn(y_pred, y_true):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


### Plotting Functions ###

def make_graph(df, outputs, inputs):
    df_selected = df[inputs + outputs]

    for val in outputs:
        fig = px.scatter_3d(
            df_selected,
            x=inputs[0],
            y=inputs[1],
            z=inputs[2],
            color=val,
            color_continuous_scale='Viridis'
            ,title=f'3D Scatter Plot with {val} as Color'  
        )

        fig.update_layout(
            scene=dict(
                xaxis_title=inputs[0],
                yaxis_title=inputs[1],
                zaxis_title=inputs[2]
            ),

            width=600,
            height=600 
        )

        fig.show()

#utility function 3: make 2D scatter plot for 2 inputs feeds + color it based on desiered output (which is z)
def plot_2d(df, xyz):
    x = df[xyz[0]]
    y = df[xyz[1]]
    z = df[xyz[2]] if xyz[2] else xyz[2]
    
    plt.figure()

    scatter = plt.scatter(x, y, c=z, cmap='viridis')

    if xyz[2]:
        scatter = plt.scatter(x, y, c=z, cmap='viridis')
        plt.colorbar(scatter, label=xyz[2])
        title = "2D Scatter Plot with " + xyz[2] + " as color (Third Dimension)"
    else:
        title = "2D Scatter Plot of " + xyz[1] + " vs " + xyz[0]
        scatter = plt.scatter(x, y)

    plt.xlabel(xyz[0])
    plt.ylabel(xyz[1])
    plt.title(title)
    plt.show()

def loss_plot(train_loss_values, test_loss_values, start=0, end=None):
    end = len(train_loss_values) if not end else end

    plt.figure(figsize=(10, 6))
    plt.plot(range(start,end), train_loss_values[start:end], label='Training Loss')
    plt.plot(range(start,end), test_loss_values[start:end], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.show()


### Linear Model Functions ###

#utility function 4: extract coeffiecnts and intercepts of Linear Regression model
def parameter_extractor(LinearModel, outputs):
    coefficients = []
    intercepts = []

    for estimator in LinearModel.estimators_:
        print(estimator) ##
        coefficients.append(estimator.coef_)
        intercepts.append(estimator.intercept_)

    inputs = LinearModel.estimators_[0].feature_names_in_

    coefficients_df = pd.DataFrame(coefficients, columns=inputs, index=outputs)
    intercepts_series = pd.Series(intercepts, index=outputs)
    
    return coefficients_df, intercepts_series

# utility function 5: train and test different models into desired inputs and outputs

def evaluate_models(df, targets, features):
    X = df[features]
    y = df[targets]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Support Vector Regressor': SVR()
    }

    results = {}
    trained_models = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        model = MultiOutputRegressor(model)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        multioutput = 'uniform_average'

        r2 = r2_score(y_test, y_pred, multioutput=multioutput)
        mse = mean_squared_error(y_test, y_pred, multioutput=multioutput)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred, multioutput=multioutput)
        MAP = 1 - np.mean(np.abs((y_test - y_pred) / y_test))

        # Store results
        results[model_name] = {
            'R2 Score': r2,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAP': MAP,
        }
        trained_models[model_name] = model
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    print('===================== Results for =====================')
    print(targets)
    print(results_df,'\n')
    return trained_models, results

#utility function 1: printing dataFrame into file
def df_file_print(df, filename="df_values.txt"):
    with open(filename, 'w') as file:
        file.write(df.to_string(index=False))
    print(f"DataFrame values saved to {filename}")


### Neural Network Functions ###

def prepare_data_kf(df, input_cols, output_cols):
    X = torch.tensor(df[input_cols].values, dtype=torch.float32)
    y = torch.tensor(df[output_cols].values, dtype=torch.float32).squeeze()

    return X, y

def prepare_data(df, input_cols, output_cols, test_size=0.2, random_state=7):
    X, y = prepare_data_kf(df, input_cols, output_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Convert to tensors
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    
    return X_train, X_test, y_train, y_test

#function to create NN model of FFNN
def makeNN(input, output,hidden_sizes, df, coefficients=pd.DataFrame([]), biases=pd.DataFrame([]), lr=0.001):
    model = FeedForwardNN(input, hidden_sizes, output, df, coefficients, biases, lr)
    
    if not ((coefficients.empty) and (biases.empty)):
        with torch.no_grad():
            # Access the first layer
            first_layer = model.layers[0]
            
            # Initialize weights and biases of the first layer only
            first_layer.weight.copy_(torch.tensor(coefficients.values, dtype=torch.float32))
            first_layer.bias.copy_(torch.tensor(biases.values, dtype=torch.float32))
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer 
    

def trainKfolds(model,criterion, optimizer, n_splits=5, shuffle=True, random_state=7, epochs=1000, freq = 0.1, printOut =1):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    X, y = prepare_data_kf(model.df, model.input, model.output)
    fold_results = []

    models = []
    all_train_losses = []
    all_val_losses = []
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        model, criterion, optimizer = makeNN(model.input, model.output, model.hidden_sizes, model.df, coefficients=model.coefficients, biases=model.biases, lr=model.lr)
        print(f'\nFold {fold + 1}/{n_splits}')

        #split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_loss_values, val_loss_values = train(
            model, criterion, optimizer, X_train, y_train, X_val, y_val,
            epochs=epochs, freq=freq, printOut=printOut
        )

        models.append(model)
        all_train_losses.append(train_loss_values)
        all_val_losses.append(val_loss_values)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val).item()
            mape = accuracy_nn(val_outputs, y_val)
            r2 = r2_score_nn(val_outputs, y_val)

        print(f'Fold {fold + 1} Validation Loss: {val_loss:.4f}, MAPE: {mape.item():.2f}%, R²: {r2.item():.4f}')

        fold_results.append({
            'fold': fold + 1,
            'val_loss': val_loss,
            'MAPE': mape.item(),
            'R²': r2.item()
        })
    
    avg_train_losses = np.mean(all_train_losses, axis=0)
    avg_val_losses = np.mean(all_val_losses, axis=0)

    # Compute average metrics
    avg_val_loss = np.mean([res['val_loss'] for res in fold_results])
    avg_mape = np.mean([res['MAPE'] for res in fold_results])
    avg_r2 = np.mean([res['R²'] for res in fold_results])

    print(f'\nAverage Validation Loss: {avg_val_loss:.4f}, Average MAPE: {avg_mape:.2f}%, Average R²: {avg_r2:.4f}')
    return models, fold_results, avg_train_losses, avg_val_losses


# function to train nn model
def train(model,criterion, optimizer, X_train, y_train, X_val, y_val, epochs=1000, freq = 0.01, printOut =1):
    train_loss_values = []
    val_loss_values = []
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass on training data
        outputs = model(X_train).squeeze()
        train_loss_nn = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        train_loss_nn.backward()
        optimizer.step()
        
        # Save training loss for plotting
        train_loss_values.append(train_loss_nn.item())
        
        # Calculate and save val loss every epoch
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss_nn = criterion(val_outputs, y_val)
            val_loss_values.append(val_loss_nn.item())

            # Print loss (epochs*freq) times
            if (epoch + 1) % int(epochs * freq) == 0:
                mape_nn = accuracy_nn(val_outputs, y_val)
                r2_nn = r2_score_nn(val_outputs, y_val)
                if printOut:
                    print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss_nn.item():.4f}, '
                          f'Validation Loss: {val_loss_nn.item():.4f}, MAPE: {mape_nn.item():.2f}%, '
                          f'R²: {r2_nn.item():.4f}')


    return train_loss_values, val_loss_values

def predict(model, X):
    model.eval()

    with torch.no_grad():
        predictions = model(X)

    return predictions.numpy()

def ensemble_predict(models, X):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(X).numpy()
            predictions.append(pred)

    return np.mean(predictions, axis=0)

def feedsYield (nnModel, cols, rows, multipleModels = 0, constExp = [0,1000,2], feedRangeBounds = [0,2000]):
    start, end, step = constExp
    feedL, feedR = feedRangeBounds
    var_col = np.arange(feedL,feedR,(feedR-feedL)/((end-start)/step))
    
    yield_results = np.full(len(cols), -1)
    yield_std = np.full(len(cols), -1)

    
    for r in np.arange(0,len(rows)):
        
        all_feeds_yield = np.array([])
        all_feeds_yield_std = np.array([])

        for i in np.arange(0,len(cols)):
            single_feed_yields_list = np.array([])

            for j in np.arange(start,end,step):
                temp = []
                const_col = np.full((int)((end-start)/step), j)

                temp = np.column_stack([const_col] * len(cols))
                temp[:, i] = var_col
                newTestData = torch.tensor(temp, dtype=torch.float32)

                predictions = ensemble_predict(nnModel,newTestData) if multipleModels else predict(nnModel, newTestData)

                b = predictions[0,r]
                
                single_feed_single_value_yield_list = (predictions[1:,r]-b)/var_col[1:]
                mean_single_feed_single_value_yield = np.mean(single_feed_single_value_yield_list)
                single_feed_yields_list = np.append(single_feed_yields_list , mean_single_feed_single_value_yield)

            single_feed_yield = np.mean(single_feed_yields_list)
            single_feed_yield_std = np.std(single_feed_yields_list)

            all_feeds_yield = np.append(all_feeds_yield, single_feed_yield)
            all_feeds_yield_std = np.append(all_feeds_yield_std, single_feed_yield_std)

        yield_results = np.vstack((yield_results, all_feeds_yield))
        yield_std = np.vstack((yield_std, all_feeds_yield_std))
        
    yield_results = yield_results[1:,:]
    yield_std = yield_std[1:,:]

    yield_results = pd.DataFrame(yield_results, columns=cols, index=rows)
    yield_std = pd.DataFrame(yield_std, columns=cols, index=rows)

    return yield_results, yield_std