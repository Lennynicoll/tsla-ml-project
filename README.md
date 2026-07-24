# Prediccion del Precio Futuro de TSLA con Machine Learning

## Descripcion del Proyecto

Sistema capaz de estimar el precio futuro de un ticker en el mercado de valores utilizando un modelo de Machine Learning supervisado basado en regresion lineal. El proyecto utiliza datos historicos de TSLA (Tesla Inc.) obtenidos a traves de la API de Alpaca Markets.

## Pipeline del Proyecto

1. **Recoleccion de Datos** -> API Alpaca (alpaca-py)
2. **Preprocesamiento** -> Ingenieria de caracteristicas, escalado
3. **EDA** -> Analisis exploratorio con graficas y estadisticas
4. **Entrenamiento** -> 8 modelos de regresion
5. **Evaluacion** -> MAE, MSE, RMSE, R
6. **Prediccion** -> Sistema para nuevos datos (.csv)

## Estructura del Repositorio

```
tsla-ml-project/
 data/
    raw/
       tsla_raw.csv
    processed/
        tsla_processed.csv
 notebooks/
    00_proyecto_completo_TSLA.ipynb   NOTEBOOK COMPLETO
    01_datos_preprocesamiento.ipynb   Persona 1
    02 EDA TSLA.ipynb                Persona 2
    notebooks/
        03_modelos_lineales_knn.ipynb  Persona 3
        tsla_modelado.ipynb           Persona 4
 docs/
    P.2 EDA TSLA.docx
 persona1_documentacion.md
 .gitignore
```

## Modelos Entrenados

| # | Modelo | Hiperparametros | R |
|---|---|---|---|
| 1 | OLS (Linear Regression) | N/A | 0.9511 |
| 2 | Ridge Regression | alpha=1.0 | 0.9510 |
| 3 | Lasso Regression | alpha=1.0 | 0.9543 |
| 4 | Bayesian Ridge | Default | 0.9512 |
| 5 | KNN Regression | k=5, weights=distance, p=1 | 0.8645 |
| 6 | Random Forest | n_estimators=100, max_depth=10 | 0.9497 |
| 7 | SVM (Linear) | kernel=linear, C=1 | **0.9574** |
| 8 | MLP (Red Neuronal) | capas=(50,), lr=0.001 | 0.9428 |

## Dataset

- **Ticker:** TSLA
- **Periodo:** Enero 2020 - Julio 2026
- **Registros:** 1,624 (despues de limpieza)
- **Features:** 11 (open, high, low, close, volume, retorno_diario, sma_5, sma_20, volatilidad_5, volumen_promedio_5, precio_anterior)
- **Variable Objetivo:** precio_futuro (precio de cierre del dia siguiente)

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
pip install -r requirements.txt
```

## Uso

1. Configurar las credenciales de Alpaca en un archivo `.env`:
   ```
   ALPACA_API_KEY=tu_api_key
   ALPACA_SECRET_KEY=tu_secret_key
   ```

2. Ejecutar el notebook completo: `notebooks/00_proyecto_completo_TSLA.ipynb`

## Resultados

El **modelo SVM con kernel lineal** obtuvo el mejor desempeno:
- **MAE:** $9.44 USD
- **RMSE:** $12.07 USD
- **R:** 0.9574

## Integrantes

- **Lenny Nicoll Nuñez Rosario Matricula:2025-1878:** Recoleccion y Preprocesamiento de Datos
- **Elmer Joel Montilla Castro Matricula: 2023-1212:** Analisis Exploratorio de Datos (EDA)
- **Edwardo Mejía Matricula: 2023-1670:** Modelos Lineales, Probabilisticos y KNN
- **Iván Hernández Matricula: 2025-1875:** Random Forest, SVM y MLP

## Licencia

Proyecto academico - Instituto Tecnologico de Las Americas (ITLA)
