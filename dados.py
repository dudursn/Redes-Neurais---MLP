import numpy as np

def normalizar_dados(data):
	for i in range(4):
		data[:,i] = (data[:,i] - min(data[:,i]))/(max(data[:,i]) - min(data[:,i]))
	return data

def pegar_dados_S():
	my_data = np.genfromtxt('treinamento.csv', delimiter=';')
	my_test = np.genfromtxt('teste.csv', delimiter=';')
	#print(len(my_test))
	my_data[0:35,4] = [0]*35
	my_data[35:70,4]= [0.5]*35
	my_data[70:,4] = [1]*35

	my_test[0:15,4] = [0]*15
	my_test[15:30,4]= [0.5]*15
	my_test[30:,4] = [1]*15

	my_data = normalizar_dados(my_data)
	my_test = normalizar_dados(my_test)
	
	X = my_data[:,:4]
	Y = my_data[:,4]
	X_teste= my_test[:,:4]
	Y_teste = my_test[:,4]
	return X, Y, X_teste, Y_teste

def pegar_dados_H():
	my_data = np.genfromtxt('treinamento.csv', delimiter=';')
	my_test = np.genfromtxt('teste.csv', delimiter=';')
	#print(len(my_test))
	my_data[0:35,4] = [-1]*35
	my_data[35:70,4]= [0]*35
	my_data[70:,4] = [1]*35

	my_test[0:15,4] = [-1]*15
	my_test[15:30,4]= [0]*15
	my_test[30:,4] = [1]*15

	my_data = normalizar_dados(my_data)
	my_test = normalizar_dados(my_test)
	
	X = my_data[:,:4]
	Y = my_data[:,4]
	X_teste= my_test[:,:4]
	Y_teste = my_test[:,4]
	return X, Y, X_teste, Y_teste

