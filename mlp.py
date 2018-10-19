# -*- coding:utf-8 -*-
import numpy as np
import dados, os
import matplotlib.pyplot as plt

EPOCAS = 10000
NUM_CAMADAS = 3
NEURONIOS_NAS_CAMADAS = [4,3,1]
ALFA = 0.1
ERRO_MINIMO = 0.03
BIAS = -1
ERRO_MEDIO = []

#Função de ativação e a sua derivada
def hiperbolica (x): 
	return np.tanh(2*x)    
def derivada_da_hiperbolica(x): 
	return 1/(np.cosh(2*x))**2 

#Pega os dados
entrada, saida_desejada, entrada_teste, saida_teste = dados.pegar_dados_H()

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
while True:
	'''
	FOWARD
	'''
	#Calcula a função de ativação com o somatorio do produto das entradas com os pesos da camada escondida
	saida_escondida = hiperbolica(np.dot(entrada, pesos_entrada_escondido) -[peso_teta_escondido]*tam_amostras_treino)    
	saida_gerada = hiperbolica(np.dot(saida_escondida, pesos_escondido_saida)-[peso_teta_saida]*tam_amostras_treino)   

	'''
	ERRO
	'''
	erro = saida_desejada - saida_gerada
	#Erro quadrático médio
	erro_medio_quadratico =np.mean((saida_gerada - saida_desejada) ** 2)
	ERRO_MEDIO.append(erro_medio_quadratico)
	if erro_medio_quadratico<=ERRO_MINIMO:
		break

	#Iteração
	i+=1
	print(i)

	'''
	BACKPROPAGATION
	'''                            
	gradiente_saida = erro * derivada_da_hiperbolica(saida_gerada)  
	gradiente_escondido = gradiente_saida.dot(pesos_escondido_saida.T) * derivada_da_hiperbolica(saida_escondida) 

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

os.system('cls')

iteracao = i  
print('Erro medio:',erro_medio_quadratico)       
print('Iteração:', iteracao)
print()

#########
#TESTE  #
#########

for i in range(len(entrada_teste)):
	saida_escondida = hiperbolica(np.dot(entrada_teste, pesos_entrada_escondido) - [peso_teta_escondido]*tam_amostras_teste)    
	saida_gerada = hiperbolica(np.dot(saida_escondida, pesos_escondido_saida)-[peso_teta_saida]*tam_amostras_teste)

#Arredondamento
saida_gerada =np.around(saida_gerada)
print(saida_gerada)
corretos = 0
for i in range(len(saida_teste)):
	if saida_gerada[i][0]==saida_teste[i]:
		corretos+=1

print('Acurácia:',(corretos/tam_amostras_teste)*100)
'''
print('Evolução do erro:\n')
plt.xlabel('Iteração')
plt.ylabel('Erro Médio')
plt.plot([x for x in range(iteracao+1)], [ERRO_MEDIO[j] for j in range(len(ERRO_MEDIO))], 'ro')

plt.show()
'''