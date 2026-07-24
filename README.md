# Prediccion del Precio Futuro de TSLA con Machine Learning

## Descripcion del Proyecto

Sistema capaz de estimar el precio futuro de un ticker en el mercado de valores utilizando un modelo de Machine Learning supervisado basado en regresion. El proyecto utiliza datos historicos de TSLA (Tesla Inc.) obtenidos a traves de la API de Alpaca Markets.

## Miembros

- **Lenny Nicoll Nunez Rosario** - Matricula: 2025-1878 - Recoleccion y Preprocesamiento de Datos
- **Elmer Joel Montilla Castro** - Matricula: 2023-1212 - Analisis Exploratorio de Datos (EDA)
- **Edwardo Mejia** - Matricula: 2023-1670 - Modelos Lineales, Probabilisticos y KNN
- **Ivan Hernandez** - Matricula: 2025-1875 - Random Forest, SVM y MLP

**Profesor:** Ramon E. Alvarez S.

---

## Pipeline del Proyecto

1. **Recoleccion de Datos** -> API Alpaca (alpaca-py)
2. **Preprocesamiento** -> Ingenieria de 11 features, escalado con StandardScaler
3. **EDA** -> Analisis exploratorio con graficas y estadisticas
4. **Entrenamiento** -> 8 modelos de regresion
5. **Evaluacion** -> MAE, MSE, RMSE, R
6. **Prediccion** -> Sistema para nuevos datos (.csv)

---

## Etapa 1: Recoleccion y Preprocesamiento de Datos

### 1.1 Fuente de Datos

Se utilizo la API de **Alpaca Markets** (libreria `alpaca-py`) para obtener el historico de precios de **TSLA**.

- **Endpoint:** `StockHistoricalDataClient` (Trading API / cuenta Paper Trading)
- **Timeframe:** Diario (`TimeFrame.Day`)
- **Rango de fechas:** 2020-01-01 al 2026-07-21
- **Registros obtenidos:** 1,644 filas (antes de limpieza)

Las credenciales estan en un archivo `.env` (no se sube a GitHub). Ver `.env.example` para el formato.

### 1.2 Features Creadas (11 en total)

| # | Feature | Descripcion |
|---|---|---|
| 1 | `open` | Precio de apertura del dia |
| 2 | `high` | Precio mas alto del dia |
| 3 | `low` | Precio mas bajo del dia |
| 4 | `close` | Precio de cierre del dia |
| 5 | `volume` | Volumen de acciones negociadas |
| 6 | `retorno_diario` | % de cambio respecto al dia anterior (`pct_change`) |
| 7 | `sma_5` | Media movil de 5 dias del precio de cierre |
| 8 | `sma_20` | Media movil de 20 dias del precio de cierre |
| 9 | `volatilidad_5` | Desviacion estandar de 5 dias del cierre |
| 10 | `volumen_promedio_5` | Media movil de 5 dias del volumen |
| 11 | `precio_anterior` | Precio de cierre del dia anterior (`shift(1)`) |

**Variable objetivo:** `precio_futuro` -> precio de cierre del dia siguiente (`shift(-1)`)

Todas las features estan escaladas con `StandardScaler` (media 0, desviacion 1). La variable objetivo se dejo sin escalar (en dolares reales).

### 1.3 Limpieza

- Se eliminaron filas con valores nulos generados por ventanas moviles y shift
- **Antes:** 1,644 filas -> **Despues:** 1,624 filas

### 1.4 Como Usar el Dataset

```python
import pandas as pd

df = pd.read_csv("data/processed/tsla_processed.csv")
X = df.drop(columns=["precio_futuro"])
y = df["precio_futuro"]
print(df.shape)  # (1624, 12)
```

---

## Etapa 2: Analisis Exploratorio de Datos (EDA)

### 2.1 Estadisticas Descriptivas

El precio futuro presento una media de US$493.29, una mediana de US$400.56 y una desviacion estandar de US$323.58.

| Variable | Media | Mediana | Desv. Estandar | Minimo | Maximo |
|---|---|---|---|---|---|
| precio_futuro | 493.289 | 400.565 | 323.582 | 108.100 | 2238.750 |
| close | -0.000 | -0.286 | 1.000 | -1.191 | 5.395 |
| retorno_diario | -0.000 | 0.005 | 1.000 | -16.140 | 4.692 |
| volume | -0.000 | -0.168 | 1.000 | -1.368 | 4.786 |
| volatilidad_5 | 0.000 | -0.201 | 1.000 | -0.400 | 19.614 |

### 2.2 Graficas Realizadas

- **Evolucion del precio futuro:** Serie temporal del precio de cierre. Permite identificar tendencias, ciclos y momentos de alta volatilidad.
- **Histograma del precio_future:** Muestra la distribucion con extension hacia valores altos.
- **Boxplot del precio_future:** El 50% central esta entre US$240.98 y US$705.64. Los valores extremos se conservaron por ser movimientos reales del mercado.
- **Histograma del retorno diario:** La mayoria se concentra cerca del promedio, con jornadas de variaciones fuertes.
- **Boxplot del volumen:** Valores extremos representan dias con actividad superior a lo habitual.
- **Matriz de correlacion:** Las relaciones mas altas con precio_futuro son close (0.9873), low (0.9860), high (0.9854) y open (0.9842).
- **Graficas de dispersion:** close, precio_anterior y sma_5 forman patrones ascendentes definidos.
- **Volatilidad y retorno:** Periodos de mayor inestabilidad coinciden con retornos mas extremos.

### 2.3 Correlacion con la Variable Objetivo

| Variable | Correlacion con precio_futuro |
|---|---|
| close | 0.9873 |
| low | 0.9860 |
| high | 0.9854 |
| open | 0.9842 |
| precio_anterior | 0.9740 |
| sma_5 | 0.9715 |
| sma_20 | 0.9170 |
| volatilidad_5 | 0.2584 |
| retorno_diario | 0.0772 |
| volume | -0.6826 |
| volumen_promedio_5 | -0.7225 |

### 2.4 Variables Mas Relevantes

Las variables close, low, high, open, precio_anterior, sma_5 y sma_20 presentan las correlaciones mas fuertes. El retorno diario y la volatilidad muestran relaciones mas debiles pero aportan informacion sobre movimientos de corto plazo. El volumen y su promedio movil muestran relacion negativa.

---

## Etapa 3: Modelos Lineales, Probabilisticos y KNN

### 3.1 Division de Datos

Se uso division cronologica estricta (no aleatoria) para respetar la serie de tiempo:

- **80% Entrenamiento:** 1,299 datos
- **20% Prueba:** 325 datos
- **Validacion cruzada:** `TimeSeriesSplit(n_splits=5)`

### 3.2 Modelos Entrenados

#### OLS (Minimos Cuadrados Ordinarios)
Punto de comparacion (baseline). R = **0.9511**, error promedio de **$9.93 USD**. Demuestra comportamiento lineal fuerte entre las features y el precio.

#### Ridge Regression
Penalizacion L2. Alpha optimo: 1.0. R = **0.9510**. Reduce la magnitud de coeficientes pero no elimina variables.

#### Lasso Regression (Mejor del bloque)
Penalizacion L1. Alpha optimo: 1.0. R = **0.9543**, MAE **$9.67 USD**. Descarta ruido e inconsistencias en los datos.

#### Bayesian Ridge
Estimacion automatica de regularizacion. R = **0.9512**, MAE **$9.92 USD**. Muy estable y robusto.

#### KNN Regression
k=5, weights=distance, p=1 (Manhattan). R = **0.8645**, MAE **$16.57 USD**. Peor rendimiento porque busca cercania espacial, no captura tendencia financiera.

---

## Etapa 4: Modelos No Lineales (Random Forest, SVM, MLP)

### 4.1 Random Forest

| n_estimators | max_depth | MAE | RMSE | R |
|---|---|---|---|---|
| 50 | 5 | 10.77 | 13.97 | 0.9429 |
| **100** | **10** | **10.11** | **13.12** | **0.9497** |
| 200 | None | 10.27 | 13.32 | 0.9482 |

**Elegido:** n_estimators=100, max_depth=10. Balancea generalizacion sin sobreajuste.

### 4.2 SVM (Support Vector Regression)

| kernel | C | MAE | RMSE | R |
|---|---|---|---|---|
| **linear** | **1** | **9.44** | **12.07** | **0.9574** |
| rbf | 1 | 27.59 | 35.79 | 0.6258 |
| rbf | 10 | 11.52 | 17.47 | 0.9108 |

**Elegido:** kernel=linear, C=1. El kernel rbf con C=1 tuvo desempeno muy inferior, confirmando que la relacion es predominantemente lineal.

### 4.3 MLP (Red Neuronal)

| hidden_layers | lr_init | MAE | RMSE | R |
|---|---|---|---|---|
| **(50,)** | **0.001** | **10.87** | **13.99** | **0.9428** |
| (100, 50) | 0.001 | 11.36 | 14.44 | 0.9390 |
| (100, 50) | 0.01 | 13.40 | 16.94 | 0.9162 |

**Elegido:** hidden_layers=(50,), lr=0.001. Arquitectura simple generalizo mejor con el tamano limitado del set de entrenamiento.

---

## Etapa 5: Evaluacion y Comparacion de los 8 Modelos

### Metricas Utilizadas

- **MAE:** Error promedio absoluto en dolares
- **RMSE:** Raiz del error cuadratico medio (penaliza errores grandes)
- **R:** Coeficiente de determinacion (1.0 = prediccion perfecta)

### Tabla Comparativa Final

| Modelo | MAE | RMSE | R | Ranking |
|---|---|---|---|---|
| **SVM Linear** | **$9.44** | **$12.07** | **0.9574** | **1** |
| Lasso | $9.67 | $12.51 | 0.9543 | 2 |
| Bayesian Ridge | $9.92 | $12.92 | 0.9512 | 3 |
| OLS | $9.93 | $12.93 | 0.9511 | 4 |
| Ridge | $10.02 | $12.95 | 0.9510 | 5 |
| Random Forest | $10.11 | $13.12 | 0.9497 | 6 |
| MLP | $10.87 | $13.99 | 0.9428 | 7 |
| KNN | $16.57 | $21.53 | 0.8645 | 8 |

### Hiperparametros y Optimizacion

| Modelo | Hiperparametros Evaluados | Valor Optimo |
|---|---|---|
| OLS | N/A | N/A |
| Ridge | alpha: 0.001, 0.01, 0.1, 1, 10, 100 | alpha=1.0 |
| Lasso | alpha: 0.001, 0.01, 0.1, 1, 10, 100 | alpha=1.0 |
| Bayesian Ridge | Configuracion por defecto | Default |
| KNN | k: 3-15, weights: uniform/distance, p: 1-2 | k=5, distance, p=1 |
| Random Forest | n_estimators: 50-200, max_depth: 5-None | n=100, depth=10 |
| SVM | kernel: linear/rbf, C: 1-10 | linear, C=1 |
| MLP | hidden: (50)-(100,50), lr: 0.001-0.01 | (50,), lr=0.001 |

---

## Etapa 6: Sistema de Prediccion con Nuevos Datos

Se implemento la funcion `predecir_precio_futuro()` que:

1. Recibe un CSV con las 11 features preprocesadas y escaladas
2. Valida que las columnas esperadas esten presentes
3. Carga el modelo SVM entrenado
4. Genera predicciones con `modelo.predict()`
5. Retorna el precio estimado del dia siguiente

**Limitacion:** Para datos reales del dia actual, es necesario replicar el pipeline completo de preprocesamiento y aplicar el mismo `StandardScaler` (no uno nuevo).

---

## Estructura del Repositorio

```
tsla-ml-project/
  data/
    raw/tsla_raw.csv
    processed/tsla_processed.csv
  notebooks/
    00_proyecto_completo_TSLA.ipynb    NOTEBOOK COMPLETO
    03_modelos_lineales_knn.ipynb      Persona 3
    tsla_modelado.ipynb                Persona 4
  docs/
    P.2 EDA TSLA.docx
    persona1_documentacion.md
    Documento_Infraestructura_TSLA_v2.docx
  .env.example
  .gitignore
  README.md
```

---

## Requisitos

```
pandas
numpy
scikit-learn
matplotlib
seaborn
alpaca-py
python-dotenv
```

## Instalacion

```bash
pip install pandas numpy scikit-learn matplotlib seaborn alpaca-py python-dotenv
```

## Uso

1. Clonar el repositorio:
```bash
git clone https://github.com/Lennynicoll/tsla-ml-project.git
```

2. Configurar credenciales en `.env`:
```
ALPACA_API_KEY=tu_api_key
ALPACA_SECRET_KEY=tu_secret_key
```

3. Ejecutar el notebook completo: `notebooks/00_proyecto_completo_TSLA.ipynb`

---

## Codigo Fuente

- **Notebook completo:** https://github.com/Lennynicoll/tsla-ml-project/blob/main/notebooks/00_proyecto_completo_TSLA.ipynb
- **Repositorio:** https://github.com/Lennynicoll/tsla-ml-project

---

Proyecto academico - Instituto Tecnologico de Las Americas (ITLA)
