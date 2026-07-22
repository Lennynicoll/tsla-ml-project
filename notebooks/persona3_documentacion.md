# Persona 3: Modelos Lineales, Probabilísticos y Vecinos Más Cercanos (KNN)

## 1. Introducción y Metodología
Para este bloque me encargué de entrenar, evaluar y ajustar los hiperparámetros de 5 modelos de regresión a partir del archivo `tsla_processed.csv` preparado por la Persona 1.

Al tratarse de una serie de tiempo (precios de acciones de TSLA), no se utilizó una división aleatoria (`shuffle=True`). En su lugar, mantuvimos el orden cronológico estricto:
* **80% Entrenamiento:** 1,299 datos.
* **20% Prueba (Test):** 325 datos.

Para la validación cruzada y el ajuste de hiperparámetros usamos `TimeSeriesSplit(n_splits=5)`, evitando así filtrar información del futuro hacia el pasado.

## 2. Resultados en el Conjunto de Prueba (Test)

* **Lasso Regression (Mejor modelo):** MAE de $9.67 USD | MSE de 156.42 | RMSE de $12.51 USD | R² de 0.9543 | Hiperparámetro: `alpha = 1.0`
* **Bayesian Ridge:** MAE de $9.92 USD | MSE de 166.97 | RMSE de $12.92 USD | R² de 0.9512 | Hiperparámetro: Configuración por defecto
* **OLS (Linear Regression):** MAE de $9.93 USD | MSE de 167.20 | RMSE de $12.93 USD | R² de 0.9511 | Hiperparámetro: N/A (Modelo base)
* **Ridge Regression:** MAE de $10.02 USD | MSE de 167.81 | RMSE de $12.95 USD | R² de 0.9510 | Hiperparámetro: `alpha = 1.0`
* **KNN Regression:** MAE de $16.57 USD | MSE de 463.73 | RMSE de $21.53 USD | R² de 0.8645 | Hiperparámetros: `n_neighbors = 5`, `weights = 'distance'`, `p = 1`

## 3. Explicación y Justificación de los Modelos

### OLS (Mínimos Cuadrados Ordinarios)
Sirve como nuestro punto de comparación (*baseline*). Consiguió un R² de **0.9511** y un error promedio de **$9.93 USD**. Demuestra que el precio de cierre tiene un comportamiento lineal muy fuerte con las variables generadas (como el precio del día anterior y las medias móviles).

### Lasso Regression (El mejor modelo del grupo)
Evaluamos el parámetro `alpha` en un rango de 0.001 a 100. El mejor resultado se obtuvo con `alpha = 1.0`. Al aplicar penalización L1, Lasso logró descartar ruido e inconsistencias en los datos, bajando el error promedio a **$9.67 USD** y alcanzando el R² más alto del bloque (**0.9543**).

### Ridge Regression
Probamos `alpha` en el mismo rango de valores. El valor óptimo fue `alpha = 1.0`. Su desempeño (R² = **0.9510**) fue muy similar al de OLS, ya que la regularización L2 reduce la magnitud de los coeficientes pero no elimina ninguna variable por completo.

### Bayesian Ridge
Este modelo estima los parámetros de regularización mediante distribuciones de probabilidad a priori. Mostró un desempeño sumamente estable (R² = **0.9512**, MAE **$9.92 USD**), ofreciendo una alternativa confiable y automática frente a OLS.

### KNN Regression (El de menor rendimiento)
Se buscaron los mejores valores con `GridSearchCV` evaluando:
* Número de vecinos (`n_neighbors` de 3 a 15).
* Ponderación (`uniform` vs. `distance`).
* Métrica de distancia (`p = 1` para Manhattan, `p = 2` para Euclidiana).

La mejor combinación fue **k = 5**, **ponderación por distancia** y **métrica Manhattan (p = 1)**. A pesar del ajuste, fue el modelo con peor rendimiento (R² = **0.8645** y error de **$16.57 USD**). Esto sucede porque KNN intenta predecir calculando cercanía espacial en el plano de características, lo cual no es tan efectivo para capturar la tendencia acumulada de una serie financiera.