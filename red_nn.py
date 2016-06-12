import math
import random

def aleatorio(a, b):
    return random.uniform(a, b)

def sigmoide(x):
    return 1 / (1 + (math.pow(math.e, -x)))  

def dsigmoide(y):
    return y * (1 - y)

def crear_matriz(f, c, llenar = 0.0):
    m = []
    for i in range(f):
        m.append([llenar] * c)
    return m
  
def matriz_aleatoria(matriz, a, b):
    for i in range ( len (matriz)):
        for j in range ( len (matriz[0]) ):
            matriz[i][j] = aleatorio(a, b)

class RedNeuronal:
    def __init__(self, Nodo_inicio, Nodo_oculto, Nodo_salida):
        ## cantidad de entradas
        self.nodo_inicio = Nodo_inicio + 1
        self.nodo_oculto = Nodo_oculto
        self.nodo_salida = Nodo_salida
    
        ## inicializar activacion de nodos
        self.activacion_inicio = [1.0] * self.nodo_inicio
        self.activacion_oculta = [1.0] * self.nodo_oculto
        self.activacion_salida = [1.0] * self.nodo_salida

        ## matriz de pesos
        self.peso_inicio = crear_matriz(self.nodo_inicio, self.nodo_oculto)
        self.peso_salida = crear_matriz(self.nodo_oculto, self.nodo_salida)
    
        ## pesos aleatorios
        matriz_aleatoria(self.peso_inicio, -0.1, 0.1)
        matriz_aleatoria(self.peso_salida, -1.0, 1.0)
    
        ## cambios en los pesos
        self.cambio_inicio = crear_matriz (self.nodo_inicio, self.nodo_oculto)
        self.cambio_salida = crear_matriz (self.nodo_oculto, self.nodo_salida)

    def correrRedNeuronal (self, entradas):
        if len(entradas) != self.nodo_inicio - 1:
            print ('numero de entradas incorrecta')

        for i in range(self.nodo_inicio-1):
            self.activacion_inicio[i] = entradas[i]

        for j in range(self.nodo_oculto):
            suma = 0.0
            for i in range(self.nodo_inicio):
                suma +=( self.activacion_inicio[i] * self.peso_inicio[i][j])
            self.activacion_oculta[j] = sigmoide(suma)
    
        for k in range(self.nodo_salida):
            suma = 0.0
            for j in range(self.nodo_oculto):        
                suma += (self.activacion_oculta[j] * self.peso_salida[j][k])
            self.activacion_salida[k] = sigmoide (suma)
      
        return self.activacion_salida

    def backPropagation(self, salidas, N, M):
        ## dE/dw[j][k] = (t[k] - ao[k]) * s'( SUM( w[j][k]*ah[j] ) ) * ah[j]
        salida_deltas = [0.0] * self.nodo_salida
        for k in range(self.nodo_salida):
            error = salidas[k] - self.activacion_salida[k]
            salida_deltas[k] =  error * dsigmoide(self.activacion_salida[k]) 
   
        ## actualizar pesos de salida
        for j in range(self.nodo_oculto):
            for k in range(self.nodo_salida):
                cambio = salida_deltas[k] * self.activacion_oculta[j]
                self.peso_salida[j][k] += N * cambio + M * self.cambio_salida[j][k]
                self.cambio_salida[j][k] = cambio

        ## calcular deltas ocultos
        oculto_deltas = [0.0] * self.nodo_oculto
        for j in range(self.nodo_oculto):
            error = 0.0
            for k in range(self.nodo_salida):
                error += salida_deltas[k] * self.peso_salida[j][k]
            oculto_deltas[j] = error * dsigmoide(self.activacion_oculta[j])
    
        ## actualizar pesos de entrada
        for i in range (self.nodo_inicio):
            for j in range (self.nodo_oculto):
                cambio = oculto_deltas[j] * self.activacion_inicio[i]
                #print('activacion', self.activacion_inicio[i], 'sinapsis', i, j,'cambio', cambio)
                self.peso_inicio[i][j] += N * cambio + M * self.cambio_inicio[i][j]
                self.cambio_inicio[i][j] = cambio
        
        ## 1/2 diferencial ** 2
        error = 0.0
        for k in range(len(salidas)):
            error += 0.5 * (salidas[k] - self.activacion_salida[k]) ** 2
        return error
    
    def pesos(self):
        print ('Entrada pesos:')
        for i in range(self.nodo_inicio):
            print (self.peso_inicio[i])
        print()
        print('Salida pesos:')
        for j in range(self.nodo_oculto):
            print (self.peso_salida[j])
        print('')
  
    def prueba(self, patrones):
        for p in patrones:
            entradas = p[0]
            print('Entrada:', p[0], '-->', self.correrRedNeuronal(entradas), '\tSalida', p[1])
  
    def entrenar (self, patrones, max_iteraciones = 1000, N = 0.5, M = 0.1):
        for i in range(max_iteraciones):
            for p in patrones:
                entradas = p[0]
                salidas = p[1]
                self.correrRedNeuronal(entradas)
                error = self.backPropagation(salidas, N, M)
            if i % 50 == 0:
                print('error', error)
        self.prueba(patrones)
    

def main ():
    pat = [
        [[0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], [0, 0, 0, 0]],
        [[0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1], [0, 0, 0, 1]],
        [[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], [0, 0, 1, 0]],
        [[1, 1, 1, 0, 0, 1, 1, 1, 1, 0 ,0, 1, 1, 1, 1], [0, 0, 1, 1]],
        [[1, 0, 1, 1, 0, 1 ,1 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,1], [0 ,1 ,0 ,0]],
        [[1 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1], [0, 1, 0, 1]],
        [[1 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1], [0 ,1 ,1 ,0]],
        [[1 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,0 ,0 ,1], [0 ,1 ,1 ,1]],
        [[1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1], [1 ,0 ,0 ,0]],
        [[1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,1], [1 ,0 ,0 ,1]],
        [[1 ,1 ,1 ,1 ,0 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,0 ,0 ,1], [0 ,1 ,1 ,1]]
    ]
  
    red1 = RedNeuronal(15, 4, 4)
    red1.pesos()
    red1.entrenar(pat)


if __name__ == "__main__":
    main()
