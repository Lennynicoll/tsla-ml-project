# Persona 3: Modelos Lineales, Probabilisticos y Vecinos Mas Cercanos (KNN)

## 1. Introduccion y Metodologia
Para este bloque me encargue de entrenar, evaluar y ajustar los hiperparametros de 5 modelos de regresion a partir del archivo `tsla_processed.csv` preparado por la Persona 1.

Al tratarse de una serie de tiempo (precios de acciones de TSLA), no se utilizo una division aleatoria (`shuffle=True`). En su lugar, mantuvimos el orden cronologico estricto:
* **80% Entrenamiento:** 1,299 datos.
* **20% Prueba (Test):** 325 datos.

Para la validacion cruzada y el ajuste de hiperparametros usamos `TimeSeriesSplit(n_splits=5)`, evitando asi filtrar informacion del futuro hacia el pasado.

## 2. Resultados en el Conjunto de Prueba (Test)

* **Lasso Regression (Mejor modelo):** MAE de $9.67 USD | MSE de 156.42 | RMSE de $12.51 USD | R de 0.9543 | Hiperparametro: `alpha = 1.0`
* **Bayesian Ridge:** MAE de $9.92 USD | MSE de 166.97 | RMSE de $12.92 USD | R de 0.9512 | Hiperparametro: Configuracion por defecto
* **OLS (Linear Regression):** MAE de $9.93 USD | MSE de 167.20 | RMSE de $12.93 USD | R de 0.9511 | Hiperparametro: N/A (Modelo base)
* **Ridge Regression:** MAE de $10.02 USD | MSE de 167.81 | RMSE de $12.95 USD | R de 0.9510 | Hiperparametro: `alpha = 1.0`
* **KNN Regression:** MAE de $16.57 USD | MSE de 463.73 | RMSE de $21.53 USD | R de 0.8645 | Hiperparametros: `n_neighbors = 5`, `weights = 'distance'`, `p = 1`

## 3. Explicacion y Justificacion de los Modelos

### OLS (Minimos Cuadrados Ordinarios)
Sirve como nuestro punto de comparacion (*baseline*). Consiguio un R de **0.9511** y un error promedio de **$9.93 USD**. Demuestra que el precio de cierre tiene un comportamiento lineal muy fuerte con las variables generadas (como el precio del dia anterior y las medias moviles).

### Lasso Regression (El mejor modelo del grupo)
Evaluamos el parametro `alpha` en un rango de 0.001 a 100. El mejor resultado se obtuvo con `alpha = 1.0`. Al aplicar penalizacion L1, Lasso logro descartar ruido e inconsistencias en los datos, bajando el error promedio a **$9.67 USD** y alcanzando el R mas alto del bloque (**0.9543**).

### Ridge Regression
Probamos `alpha` en el mismo rango de valores. El valor optimo fue `alpha = 1.0`. Su desempeno (R = **0.9510**) fue muy similar al de OLS, ya que la regularizacion L2 reduce la magnitud de los coeficientes pero no elimina ninguna variable por completo.

### Bayesian Ridge
Este modelo estima los parametros de regularizacion mediante distribuciones de probabilidad a priori. Mostro un desempeno sumamente estable (R = **0.9512**, MAE **$9.92 USD**), ofreciendo una alternativa confiable y automatica frente a OLS.

### KNN Regression (El de menor rendimiento)
Se buscaron los mejores valores con `GridSearchCV` evaluando:
* Numero de vecinos (`n_neighbors` de 3 a 15).
* Ponderacion (`uniform` vs. `distance`).
* Metrica de distancia (`p = 1` para Manhattan, `p = 2` para Euclidiana).

La mejor combinacion fue **k = 5**, **ponderacion por distancia** y **metrica Manhattan (p = 1)**. A pesar del ajuste, fue el modelo con peor rendimiento (R = **0.8645** y error de **$16.57 USD**). Esto sucede porque KNN intenta predecir calculando cercania espacial en el plano de caracteristicas, lo cual no es tan efectivo para capturar la tendencia acumulada de una serie financiera.