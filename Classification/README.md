
Instructions to run the classification model

Dependencies

1.Numpy
2.Tensorflow
	

1. Open the classification folder and make sure there are 3 files loader2.py , lstm_network.py and test.py

2. Three kinds of hyper-parameters

   Data Hyper - Parameters 

	NUM_STEPS - (Number of days used to make prediction)

	LEAD_TIME - (Number of days ahead to make the prediction)

	To change data hyper parameters , change the values of NUM_STEPS or LEAD_TIME in loader2.py to corresponding values.

	
  

   Model Hyper-Parameters


	lstm_size - (Size of LSTM cell at each layer)
	
	Example :-
	If only one layer - lstm_size = [40]

	If two layers - [40 , 80]

	For n layers - [30 , 50 , ..... 100] - integer list of size n


	dropout - (Dropout value at each layer (Values between 0 and 1))
	If only one layer - lstm_size = [0.8]

	If two layers - [0.8 , 0.6]

	For n layers - [0.8 , 0.9 , 1 ...... 0.5] -  list of size n

	Length of dropout list should be same as length of lstm_size list



	
	hidden_layers - Number of elements in list of lstm_size(No need to change this value , only need to change lstm_size)

	These parameters value can be changed in lstm_network.py in the class RNNConfig().
	

	IMPORTANT
	If lstm_size is changed in lstm_network.py then the same change has to done in test.py with same value.


  Learning Hyper-Parameters

    init_learning_rate  - Sets the initial learning rate for the model

    learning_rate_decay = 0.99 - By how much the learning rate will decay over time

    lamda = 0.001 - Regularization parameter

    max_epoch = 100  - Total number of epochs

	

    These hyper-parameters can be changed in RNNConfig of lstm_network.py



3. Variables used

	The variables used can be found in loader2.py

	The variables used are in form of lists

	uwind = [uwindCI , uwindBOB , uwindAS , uwindSI]
	vwind = [vwindCI , vwindBOB , vwindAS , vwindSI]
	at = [atCI , atBOB , atAS , atSI]
	pres = [presCI , presBOB , presAS , presSI]

	To remove any variable remove it from the list and correspondingly change value of INPUTS in loader.py



4. Running the model


	To run the model , run the file python lstm_network.py

	Model saves it's state every 10 steps in the saved_networks folder

	Program can be paused at any time and can resume training from previously saved checkpoint.
	
	To test the model at any instant , stop the lstm_network.py program and then run python test.py


	Everytime any Model hyperparameter or Data Hyper-Parameter is changed , the saved_networks folder has to be deleted so that the model 		can start off with new hyperparameters.
	

	


		

 
