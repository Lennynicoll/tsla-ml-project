# Documentacion - Modelado (Persona 4)

**Responsable:** Ivan Hernandez
**Notebook asociado:** `tsla_modelado.ipynb` (modelado: Random Forest, SVM, MLP)

## Objetivo de esta etapa

Entrenar y comparar tres modelos de regresion -Random Forest, SVM y una red
neuronal (MLP)- para predecir el precio de cierre de Tesla del dia siguiente,
usando el dataset ya preprocesado por el equipo (`tsla_processed.csv`), y
dejar listo el modelo con mejor desempeno para generar predicciones sobre
datos nuevos.

## 1. Division de entrenamiento y prueba

Se uso una proporcion **80% entrenamiento / 20% prueba**, respetando el
**orden cronologico** de las filas (no aleatorio), ya que se trata de una
serie de tiempo. El primer 80% de los registros (mas antiguos) se uso para
entrenar; el ultimo 20% (mas reciente) se reservo para evaluar, evitando asi
que el modelo "viera" informacion del futuro durante el entrenamiento.

- Filas de entrenamiento: 1,299
- Filas de prueba: 325

## 2. Descripcion de los algoritmos utilizados

| Modelo | Descripcion |
|---|---|
| **Random Forest** | Ensemble de multiples arboles de decision; cada arbol aprende reglas de division sobre las features y la prediccion final se obtiene promediando las predicciones de todos los arboles. |
| **SVM (Support Vector Regression)** | Busca la funcion (lineal o no lineal, segun el kernel) que mejor se ajusta a los datos dentro de un margen de tolerancia, minimizando el error fuera de ese margen. |
| **MLP (Multi-Layer Perceptron)** | Red neuronal totalmente conectada; ajusta pesos internos entre capas mediante retropropagacion para minimizar el error de prediccion. |

## 3. Hiperparametros evaluados por modelo

### Random Forest
| n_estimators | max_depth | MAE | RMSE | R |
|---|---|---|---|---|
| 50 | 5 | 10.7699 | 13.9743 | 0.9429 |
| **100** | **10** | **10.1116** | **13.1188** | **0.9497** |
| 200 | None | 10.2738 | 13.3182 | 0.9482 |

**Elegido:** `n_estimators=100, max_depth=10`. Balancea capacidad de
generalizacion sin caer en sobreajuste (observado con `max_depth=None`, donde
el error vuelve a subir levemente).

### SVM (SVR)
| kernel | C | MAE | RMSE | R |
|---|---|---|---|---|
| **linear** | **1** | **9.4435** | **12.0678** | **0.9574** |
| rbf | 1 | 27.5850 | 35.7853 | 0.6258 |
| rbf | 10 | 11.5163 | 17.4688 | 0.9108 |

**Elegido:** `kernel=linear, C=1`. Fue el mejor modelo entre los tres
algoritmos evaluados. El desempeno muy inferior del kernel `rbf` con `C=1`
respalda la conclusion de que la relacion entre las variables de entrada y
el precio futuro es predominantemente lineal (consistente con la matriz de
correlacion obtenida en el EDA, donde `close`, `high`, `low`, `open`
correlacionan >0.98 con `precio_futuro`).

### MLP (red neuronal)
| hidden_layer_sizes | learning_rate_init | MAE | RMSE | R |
|---|---|---|---|---|
| **(50,)** | **0.001** | **10.8735** | **13.9901** | **0.9428** |
| (100, 50) | 0.001 | 11.3645 | 14.4447 | 0.9390 |
| (100, 50) | 0.01 | 13.4016 | 16.9383 | 0.9162 |

**Elegido:** `hidden_layer_sizes=(50,), learning_rate_init=0.001`. Una
arquitectura mas simple generalizo mejor dado el tamano limitado del set de
entrenamiento (1,299 filas); arquitecturas mas profundas mostraron
desempeno inferior.

## 4. Metricas y estadisticas de rendimiento

| Modelo | Mejor configuracion | MAE | RMSE | R |
|---|---|---|---|---|
| Random Forest | n_estimators=100, max_depth=10 | 10.1116 | 13.1188 | 0.9497 |
| **SVM (ganador)** | kernel=linear, C=1 | **9.4435** | **12.0678** | **0.9574** |
| MLP | capas=(50,), lr=0.001 | 10.8735 | 13.9901 | 0.9428 |

**Modelo seleccionado para produccion:** SVM con kernel lineal, por
presentar el menor error (MAE, RMSE) y el mayor coeficiente de
determinacion (R) de los tres algoritmos evaluados.

## 5. Explicacion del funcionamiento del sistema de prediccion

Con el modelo SVM lineal ya entrenado sobre los datos de entrenamiento, se
implemento la funcion `predecir_precio_futuro()`, que:

1. Recibe la ruta de un archivo CSV con datos nuevos (ya preprocesados y
   escalados con el mismo `StandardScaler` utilizado en la etapa de
   preprocesamiento).
2. Valida que el archivo contenga las columnas esperadas por el modelo; si
   falta alguna, lanza un error explicito en lugar de fallar de forma
   ambigua.
3. Genera la prediccion del precio futuro para cada fila mediante
   `modelo.predict()`.

La funcion fue validada con una muestra de 5 filas del propio conjunto de
prueba (`X_test`), guardadas como `prueba.csv`, comparando la prediccion
contra el valor real correspondiente. Las diferencias obtenidas estuvieron
en el rango de $10-20 dolares, coherentes con el MAE general del modelo
(9.44).

El notebook incluye, ademas, un bloque de ejemplo (comentado) que ilustra
como se usaria la funcion con un archivo de datos nuevos real
(`datos_nuevos.csv`) una vez que el equipo lo tenga disponible.

**Limitacion conocida:** para evaluar el modelo con datos de mercado
reales del dia actual, es necesario replicar el pipeline completo de
preprocesamiento (calculo de medias moviles, retorno diario, volatilidad,
etc.) y aplicar el mismo `StandardScaler` ajustado sobre los datos de
entrenamiento originales - no un scaler nuevo entrenado con datos parciales.
Este componente queda pendiente de integracion con el resto del equipo.
