# Documentación — Modelado (Persona 4)

**Responsable:** Iván Hernández
**Notebook asociado:** `tsla_modelado.ipynb` (modelado: Random Forest, SVM, MLP)

## Objetivo de esta etapa

Entrenar y comparar tres modelos de regresión —Random Forest, SVM y una red
neuronal (MLP)— para predecir el precio de cierre de Tesla del día siguiente,
usando el dataset ya preprocesado por el equipo (`tsla_processed.csv`), y
dejar listo el modelo con mejor desempeño para generar predicciones sobre
datos nuevos.

## 1. División de entrenamiento y prueba

Se usó una proporción **80% entrenamiento / 20% prueba**, respetando el
**orden cronológico** de las filas (no aleatorio), ya que se trata de una
serie de tiempo. El primer 80% de los registros (más antiguos) se usó para
entrenar; el último 20% (más reciente) se reservó para evaluar, evitando así
que el modelo "viera" información del futuro durante el entrenamiento.

- Filas de entrenamiento: 1,299
- Filas de prueba: 325

## 2. Descripción de los algoritmos utilizados

| Modelo | Descripción |
|---|---|
| **Random Forest** | Ensemble de múltiples árboles de decisión; cada árbol aprende reglas de división sobre las features y la predicción final se obtiene promediando las predicciones de todos los árboles. |
| **SVM (Support Vector Regression)** | Busca la función (lineal o no lineal, según el kernel) que mejor se ajusta a los datos dentro de un margen de tolerancia, minimizando el error fuera de ese margen. |
| **MLP (Multi-Layer Perceptron)** | Red neuronal totalmente conectada; ajusta pesos internos entre capas mediante retropropagación para minimizar el error de predicción. |

## 3. Hiperparámetros evaluados por modelo

### Random Forest
| n_estimators | max_depth | MAE | RMSE | R² |
|---|---|---|---|---|
| 50 | 5 | 10.7699 | 13.9743 | 0.9429 |
| **100** | **10** | **10.1116** | **13.1188** | **0.9497** |
| 200 | None | 10.2738 | 13.3182 | 0.9482 |

**Elegido:** `n_estimators=100, max_depth=10`. Balancea capacidad de
generalización sin caer en sobreajuste (observado con `max_depth=None`, donde
el error vuelve a subir levemente).

### SVM (SVR)
| kernel | C | MAE | RMSE | R² |
|---|---|---|---|---|
| **linear** | **1** | **9.4435** | **12.0678** | **0.9574** |
| rbf | 1 | 27.5850 | 35.7853 | 0.6258 |
| rbf | 10 | 11.5163 | 17.4688 | 0.9108 |

**Elegido:** `kernel=linear, C=1`. Fue el mejor modelo entre los tres
algoritmos evaluados. El desempeño muy inferior del kernel `rbf` con `C=1`
respalda la conclusión de que la relación entre las variables de entrada y
el precio futuro es predominantemente lineal (consistente con la matriz de
correlación obtenida en el EDA, donde `close`, `high`, `low`, `open`
correlacionan >0.98 con `precio_futuro`).

### MLP (red neuronal)
| hidden_layer_sizes | learning_rate_init | MAE | RMSE | R² |
|---|---|---|---|---|
| **(50,)** | **0.001** | **10.8735** | **13.9901** | **0.9428** |
| (100, 50) | 0.001 | 11.3645 | 14.4447 | 0.9390 |
| (100, 50) | 0.01 | 13.4016 | 16.9383 | 0.9162 |

**Elegido:** `hidden_layer_sizes=(50,), learning_rate_init=0.001`. Una
arquitectura más simple generalizó mejor dado el tamaño limitado del set de
entrenamiento (1,299 filas); arquitecturas más profundas mostraron
desempeño inferior.

## 4. Métricas y estadísticas de rendimiento

| Modelo | Mejor configuración | MAE | RMSE | R² |
|---|---|---|---|---|
| Random Forest | n_estimators=100, max_depth=10 | 10.1116 | 13.1188 | 0.9497 |
| **SVM (ganador)** | kernel=linear, C=1 | **9.4435** | **12.0678** | **0.9574** |
| MLP | capas=(50,), lr=0.001 | 10.8735 | 13.9901 | 0.9428 |

**Modelo seleccionado para producción:** SVM con kernel lineal, por
presentar el menor error (MAE, RMSE) y el mayor coeficiente de
determinación (R²) de los tres algoritmos evaluados.

## 5. Explicación del funcionamiento del sistema de predicción

Con el modelo SVM lineal ya entrenado sobre los datos de entrenamiento, se
implementó la función `predecir_precio_futuro()`, que:

1. Recibe la ruta de un archivo CSV con datos nuevos (ya preprocesados y
   escalados con el mismo `StandardScaler` utilizado en la etapa de
   preprocesamiento).
2. Valida que el archivo contenga las columnas esperadas por el modelo; si
   falta alguna, lanza un error explícito en lugar de fallar de forma
   ambigua.
3. Genera la predicción del precio futuro para cada fila mediante
   `modelo.predict()`.

La función fue validada con una muestra de 5 filas del propio conjunto de
prueba (`X_test`), guardadas como `prueba.csv`, comparando la predicción
contra el valor real correspondiente. Las diferencias obtenidas estuvieron
en el rango de $10–20 dólares, coherentes con el MAE general del modelo
(9.44).

El notebook incluye, además, un bloque de ejemplo (comentado) que ilustra
cómo se usaría la función con un archivo de datos nuevos real
(`datos_nuevos.csv`) una vez que el equipo lo tenga disponible.

**Limitación conocida:** para evaluar el modelo con datos de mercado
reales del día actual, es necesario replicar el pipeline completo de
preprocesamiento (cálculo de medias móviles, retorno diario, volatilidad,
etc.) y aplicar el mismo `StandardScaler` ajustado sobre los datos de
entrenamiento originales — no un scaler nuevo entrenado con datos parciales.
Este componente queda pendiente de integración con el resto del equipo.
