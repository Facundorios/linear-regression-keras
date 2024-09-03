import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Lectura de los Datos
df = pd.read_csv('altura_peso.csv')

# 2. Procesamiento de Datos
x = df['Altura'].values
y = df['Peso'].values

# Convertir las variables a formato adecuado para Keras
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# 3. Implementación del Modelo de Regresión Lineal con Keras
model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))

# 4. Compilación del Modelo
model.compile(optimizer='adam', loss='mse')

# 5. Entrenamiento del Modelo
model.fit(x, y, epochs=100, verbose=1)

# 6. Visualización de Resultados
plt.scatter(x, y, color='blue', label='Datos Originales')
plt.plot(x, model.predict(x), color='red', label='Línea de Regresión')
plt.title('Regresión Lineal de Altura vs Peso')
plt.xlabel('Altura (cm)')
plt.ylabel('Peso (kg)')
plt.legend()
plt.show()

# 7. Predicción de peso basada en una altura de ejemplo
altura_ejemplo = np.array([[170]])  # Altura de 170 cm
peso_predicho = model.predict(altura_ejemplo)
print(f"El peso predicho para una altura de {altura_ejemplo[0][0]} cm es {peso_predicho[0][0]:.2f} kg")
