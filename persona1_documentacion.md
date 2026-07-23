# Documentacion - Etapa 1: Recoleccion y Preprocesamiento de Datos
**Responsable:** Persona 1
**Proyecto:** Prediccion de precio de TSLA con Regresion Lineal
**Ultima actualizacion:** 21/07/2026

---

## 1. Fuente de datos

Se utilizo la API de **Alpaca Markets** (libreria `alpaca-py`) para obtener el historico de precios de **TSLA**.

- **Endpoint usado:** `StockHistoricalDataClient` (Trading API / cuenta Paper Trading)
- **Timeframe:** Diario (`TimeFrame.Day`)
- **Rango de fechas:** 2020-01-01 -> 2026-07-21
- **Registros obtenidos:** 1,644 filas (antes de limpieza)

Las credenciales (API Key y Secret Key) estan guardadas en un archivo `.env` en la raiz del proyecto y **no se suben a GitHub** (estan protegidas en `.gitignore`). Si necesitas correr el notebook desde cero, pide las keys por separado o genera las tuyas propias en [alpaca.markets](https://alpaca.markets).

---

## 2. Estructura del repositorio

```
tsla-ml-project/
 data/
    raw/
       tsla_raw.csv            historico crudo de Alpaca
    processed/
        tsla_processed.csv      dataset limpio y listo para usar
 notebooks/
    01_datos_preprocesamiento.ipynb    codigo completo de esta etapa
 docs/
 .env                             (no esta en GitHub, cada quien crea el suyo)
 .gitignore
```

**El archivo que todos deben usar es:** `data/processed/tsla_processed.csv`

---

## 3. Features creadas (11 en total)

| # | Columna | Descripcion |
|---|---|---|
| 1 | `open` | Precio de apertura del dia |
| 2 | `high` | Precio mas alto del dia |
| 3 | `low` | Precio mas bajo del dia |
| 4 | `close` | Precio de cierre del dia |
| 5 | `volume` | Volumen de acciones negociadas |
| 6 | `retorno_diario` | % de cambio respecto al dia anterior (`pct_change`) |
| 7 | `sma_5` | Media movil de 5 dias del precio de cierre |
| 8 | `sma_20` | Media movil de 20 dias del precio de cierre |
| 9 | `volatilidad_5` | Desviacion estandar de 5 dias (medida de riesgo/inestabilidad) |
| 10 | `volumen_promedio_5` | Media movil de 5 dias del volumen |
| 11 | `precio_anterior` | Precio de cierre del dia anterior (`shift(1)`) |

**Variable objetivo (target):**
- `precio_futuro` -> precio de cierre del **dia siguiente** (`shift(-1)`)

 Todas las features (columnas 1-11) estan **escaladas** con `StandardScaler` de scikit-learn (media 0, desviacion estandar 1). La columna `precio_futuro` se dejo **sin escalar**, en su valor real en dolares, porque es lo que se va a predecir.

---

## 4. Limpieza realizada

- Se eliminaron filas con valores nulos (`NaN`), generados por las ventanas moviles (`sma_20` necesita 20 dias previos) y por el corrimiento (`shift`) al inicio/final del dataset.
- **Antes de limpiar:** 1,644 filas
- **Despues de limpiar:** 1,624 filas
- Cumple de sobra el requisito de 800+ registros pedido por el profesor.

---

## 5. Como usar el dataset (para Persona 2, 3 y 4)

```python
import pandas as pd

df = pd.read_csv("../data/processed/tsla_processed.csv")

# Separar features (X) y variable objetivo (y)
X = df.drop(columns=["precio_futuro"])
y = df["precio_futuro"]

print(df.shape)   # (1624, 12)
df.head()
```

- **Persona 2 (EDA):** puede usar `df` directamente para graficas, estadisticas descriptivas y matriz de correlacion.
- **Persona 3 y 4 (Modelos):** ya tienen `X` (11 features escaladas) y `y` (precio futuro) listos para hacer el split `train_test_split` y entrenar sus modelos - **no hace falta volver a escalar ni limpiar nada.**

---

## 6. Pendiente / proximos pasos (equipo)

- [ ] Split train/test (sugerido: 80/20) - se puede definir en conjunto para que todos usen la misma division.
- [ ] EDA completo con graficas (Persona 2)
- [ ] Entrenamiento de los 8 modelos (Persona 3 y 4)
- [ ] Tabla comparativa de metricas (MAE, MSE, RMSE, R)
- [ ] Funcion de prediccion con nuevo `.csv`
- [ ] Documento de infraestructura final (unir las 4 partes)

---

## 7. Notas tecnicas para la exposicion

- El uso de `StandardScaler` se justifica porque los modelos como SVM, KNN y redes neuronales (MLP) son sensibles a la escala de las variables - sin esto, columnas como `volume` (millones) dominarian sobre columnas como `retorno_diario` (decimales).
- El uso de medias moviles (`sma_5`, `sma_20`) y volatilidad son indicadores tecnicos comunes en analisis financiero, y ayudan al modelo a capturar tendencias y riesgo, no solo el precio puntual de un dia.
- La variable objetivo se definio como el precio del **dia siguiente** (horizonte de prediccion de 1 dia) - esto debe explicarse claramente en la exposicion ya que define que tipo de prediccion hace el sistema.
