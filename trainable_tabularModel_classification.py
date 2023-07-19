import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils_data
import torch.nn.functional as F
from torch.autograd import Variable
import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score


from activation_fun import *
from performance import *
from Classification_TabularModel import *



from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
smote = SMOTE() #smote.fit_resample(X, y)
ros = RandomOverSampler() #ros.fit_resample(X, y)
adasyn = ADASYN() #adasyn.fit_resample(X, y)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Define the folder path where the CSV files are located
folder_path = 'data/imbalanced/'

# Get the list of CSV files in the folder
data_files = [file for file in os.listdir(folder_path) if file.endswith('pima.csv')]

for file in data_files:
    # Load and preprocess your tabular data from the CSV file
    csv_path = os.path.join(folder_path, file)
    data = pd.read_csv(csv_path)

    # Normalize the continuous variables using min-max normalization
    continuous_columns = data.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    data[continuous_columns] = scaler.fit_transform(data[continuous_columns])

    # Perform label encoding on categorical columns
    categorical_columns = data.select_dtypes(include='object').columns
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])

    csv_result_name = f'TabNet_{file}'

    # Fill missing values
    data = data.fillna(0)

    # Split the data into features x and labels y
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the numpy arrays to PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)


    # Build and train the models
    input_dim = train_data.shape[1]  # Define the input dimension based on my data
    hidden_dim = 256  # Define the size of the hidden layer
    output_dim = 2  # Define the number of output classes
    batchsize = 128
    Learning_Rate = 0.001
    num_epochs = 1000 # Adjust the number of epochs as needed

    training_samples = utils_data.TensorDataset(train_data, train_labels)
    data_loader = utils_data.DataLoader(training_samples, batch_size=batchsize, shuffle=False)	


    # Add activation functions here
    activation_functions = [CosLU, DELU, ReLUN, ScaledSoftSign, ShiLU, HELU, SinLU,
                            nn.ReLU, nn.GELU, nn.ELU]  


    result = {'Activation Func':[],'G mean': [], 'AUC': [], 'ACC': [], 'brier_score':[]}
    results = {}
    accuracies = []
    losses = []

    for activation_fn in activation_functions:
        
        # Build the model
        model = TabNet_decoder_Model(input_dim, hidden_dim, output_dim, activation_fn())
        print(model)   

        # Train the model
        #criterion = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.8)
        optimizer = optim.Adam(model.parameters(), lr = Learning_Rate)#, momentum=0.8)


        epoch_losses = []
        epoch_accuracies = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            total_samples = 0
            correct_predictions = 0
            
            for batch_idx, (batch_data, batch_labels) in enumerate(data_loader):
                batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
                optimizer.zero_grad()
                output = model(batch_data)
                loss = criterion(output, batch_labels)
                #print(batch_idx, loss.data[0])
                loss.backward()
                optimizer.step()
                # Track the loss and accuracy
                epoch_loss += loss.item() * len(batch_labels)
                _, predicted_labels = torch.max(output.data, 1)
                total_samples += len(batch_labels)
                correct_predictions += (predicted_labels == batch_labels).sum().item()
            epoch_loss /= total_samples
            epoch_losses.append(epoch_loss)
        
                

            # Evaluate the model
            with torch.no_grad():
                model.eval()
                test_outputs  = model(test_data)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = (predicted == test_labels).sum().item() / len(test_labels)
                auc_score,  acc, gmean, brier_score = evaluation(test_labels, predicted) #f_score_micro, f_score_macro, f_score_weighted,

                y_scores = model(test_data)
                y_scores = nn.Softmax(dim=1)(y_scores)[:, 1]  # Take the probability for the positive class

                
                result['G mean'].append(gmean)
                result['AUC'].append(auc_score)
                result['ACC'].append(acc)
                result['brier_score'].append(brier_score)
                result['Activation Func'].append(activation_fn.__name__)
                #epoch_accuracies.append(accuracy)



        results[activation_fn.__name__] = accuracy
        losses.append(epoch_losses)
        #accuracies.append(epoch_accuracies)

    # Calculate the FPR, TPR, and thresholds
        fpr, tpr, thresholds = roc_curve(test_labels, y_scores)

        # Calculate the AUC-ROC score
        auc_score = roc_auc_score(test_labels, y_scores)

        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f'{activation_fn.__name__} (AUC = {auc_score:.2f})')

    # Plot the ROC curve
    plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    # Save the ROC curve as an image
    plt.savefig(f'result_image/{csv_result_name}_roc_curve.png')

    # Plot the loss
    plt.figure(figsize=(10, 6))
    for i, loss in enumerate(losses):
        plt.plot(range(1, num_epochs + 1), loss, label=activation_functions[i].__name__)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss for Different Activation Functions')
    plt.legend(loc='upper right')
    # Save the ROC curve as an image
    plt.savefig(f'result_image/{csv_result_name}_Loss for Different Activation Functions.png')



    Accuracy_all = {'Activation Func':[],'ACC': []}    
    # Compare and print the results
    print("Activation Function Comparison:")
    for activation_fn, accuracy in results.items():
        print(f"{activation_fn}: Accuracy = {accuracy:.4f}")
        Accuracy_all['Activation Func'].append(activation_fn)
        Accuracy_all['ACC'].append(accuracy)



    ACC = pd.DataFrame.from_dict(Accuracy_all)
    ACC.to_csv(f'result/{csv_result_name}_Accuracy_all.csv', index=False)


    res = pd.DataFrame.from_dict(result)
    res.to_csv(f'result/{csv_result_name}_result.csv', index=False)