# -*- coding:utf-8 -*-
import numpy as np
import dados, os
import matplotlib.pyplot as plt

EPOCAS = 10000
NUM_CAMADAS = 3
NEURONIOS_NAS_CAMADAS = [4,3,1]
ALFA = 0.1
ERRO_MINIMO = 0.01
BIAS = -1
ERRO_MEDIO = []

#Função de ativação e a sua derivada
def sigmoid (x): 
	return 1 / (1 + np.exp(-x))   
def derivada_da_sigmoid(x): 
	return np.multiply(x, (1-x))

#Verificar o quão próximo está da classe que foi atribuida com 0.5
def verifica_saida(saida):
	nova_saida = []
	for e in saida:
		dif1 =abs(0-e)
		dif2 = abs(0.5-e)
		dif3 = abs(1-e)
		menor = min(dif1,dif2, dif3)
		if menor == dif1:
			nova_saida.append(0)

		elif menor == dif2:
			nova_saida.append(0.5)
		else:
			nova_saida.append(1)

	return np.array(nova_saida)



#Pega os dados
entrada, saida_desejada, entrada_teste, saida_teste = dados.pegar_dados_S()

#Pega o tamanho da amostra de treino e de teste
tam_amostras_treino = len(entrada)
tam_amostras_teste = len(entrada_teste)

#Transforma o vetor das saidas desejadas em uma matriz
saida_desejada = np.reshape(saida_desejada,(tam_amostras_treino,1))


#Cria uma matriz de pesos aleatorios entre a camada de entrada e a camada escondida, no caso (4 x 3)
pesos_entrada_escondido = np.random.uniform(size=(NEURONIOS_NAS_CAMADAS[0], NEURONIOS_NAS_CAMADAS[1]))

#Cria uma matriz de pesos aleatorios entre a camada escondida e a camada de saida, no caso (3 x 1)
pesos_escondido_saida = np.random.uniform(size=(NEURONIOS_NAS_CAMADAS[1], NEURONIOS_NAS_CAMADAS[2]))
peso_teta_escondido = np.random.uniform(size=NEURONIOS_NAS_CAMADAS[1])
peso_teta_saida = np.random.uniform(size=NEURONIOS_NAS_CAMADAS[2])

i = 0

##############
#TREINAMANETO#
##############
while i<=EPOCAS:
	'''
	FOWARD
	'''
	#Calcula a função de ativação com o somatorio do produto das entradas com os pesos da camada escondida
	saida_escondida = sigmoid(np.dot(entrada, pesos_entrada_escondido) -[peso_teta_escondido]*tam_amostras_treino)    
	saida_gerada = sigmoid(np.dot(saida_escondida, pesos_escondido_saida)-[peso_teta_saida]*tam_amostras_treino)   

	'''
	ERRO
	'''

	erro = saida_desejada - saida_gerada
	#Erro quadrático médio
	erro_medio_quadratico =np.mean((saida_gerada - saida_desejada) ** 2)
	ERRO_MEDIO.append(erro_medio_quadratico)
	if erro_medio_quadratico<=ERRO_MINIMO:
		break
	'''
	erro = saida_desejada - saida_gerada
	erro_copia = erro.copy()

	erro_copia = np.absolute(erro_copia)
	
	erro_max = np.max(erro_copia)
	if erro_max<=ERRO_MINIMO:
		break
	'''
	#Iteração
	i+=1
	print(i)
	
	'''
	BACKPROPAGATION
	'''            
              
	gradiente_saida = erro * derivada_da_sigmoid(saida_gerada)  
	gradiente_escondido = gradiente_saida.dot(pesos_escondido_saida.T) * derivada_da_sigmoid(saida_escondida) 

	'''
	CORREÇÃO DE PESOS    
	'''
	delta_pesos_escondido_saida = ALFA*saida_escondida.T.dot(gradiente_saida)
	delta_peso_teta_saida = ALFA*gradiente_saida.T.dot([BIAS]*tam_amostras_treino)
	delta_pesos_entrada_escondido = ALFA*entrada.T.dot(gradiente_escondido)
	delta_peso_teta_escondido  = ALFA*gradiente_escondido.T.dot([BIAS]*tam_amostras_treino)

	'''
	ATUALIZAÇÃO DOS PESOS
	'''        
	pesos_escondido_saida +=  delta_pesos_escondido_saida            
	pesos_entrada_escondido += delta_pesos_entrada_escondido   
	peso_teta_escondido +=delta_peso_teta_escondido
	peso_teta_saida += delta_peso_teta_saida

#Informações
#print(erro_medio_quadratico)
print(saida_gerada)	     
#os.system('cls')  
iteracao = i  
print('Erro medio:',erro_medio_quadratico)       
print('Iteração:', iteracao)
print()

#########
#TESTE  #
#########
input()
for i in range(len(entrada_teste)):
	saida_escondida = sigmoid(np.dot(entrada_teste, pesos_entrada_escondido) - [peso_teta_escondido]*tam_amostras_teste)    
	saida_gerada = sigmoid(np.dot(saida_escondida, pesos_escondido_saida)-[peso_teta_saida]*tam_amostras_teste)

#Arredondamento
saida_gerada =list(np.around(saida_gerada[0:15]))+list(verifica_saida(saida_gerada[15:30]))+list(np.around(saida_gerada[30:45]))
print(saida_gerada)
corretos = 0
for i in range(len(saida_teste)):
	if saida_gerada[i]==saida_teste[i]:
		corretos+=1

print('Acurácia:',(corretos/tam_amostras_teste)*100)

##########
#GRÁFICO #
##########
print('Evolução do erro:\n')
plt.xlabel('Iteração')
plt.ylabel('Erro Médio')
plt.plot([x for x in range(iteracao+1)], [ERRO_MEDIO[j] for j in range(len(ERRO_MEDIO))], 'ro')

plt.show()
