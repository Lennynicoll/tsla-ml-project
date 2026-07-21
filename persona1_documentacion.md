# Documentación — Etapa 1: Recolección y Preprocesamiento de Datos
**Responsable:** Persona 1
**Proyecto:** Predicción de precio de TSLA con Regresión Lineal
**Última actualización:** 21/07/2026

---

## 1. Fuente de datos

Se utilizó la API de **Alpaca Markets** (librería `alpaca-py`) para obtener el histórico de precios de **TSLA**.

- **Endpoint usado:** `StockHistoricalDataClient` (Trading API / cuenta Paper Trading)
- **Timeframe:** Diario (`TimeFrame.Day`)
- **Rango de fechas:** 2020-01-01 → 2026-07-21
- **Registros obtenidos:** 1,644 filas (antes de limpieza)

Las credenciales (API Key y Secret Key) están guardadas en un archivo `.env` en la raíz del proyecto y **no se suben a GitHub** (están protegidas en `.gitignore`). Si necesitas correr el notebook desde cero, pide las keys por separado o genera las tuyas propias en [alpaca.markets](https://alpaca.markets).

---

## 2. Estructura del repositorio

```
tsla-ml-project/
├── data/
│   ├── raw/
│   │   └── tsla_raw.csv           ← histórico crudo de Alpaca
│   └── processed/
│       └── tsla_processed.csv     ← dataset limpio y listo para usar
├── notebooks/
│   └── 01_datos_preprocesamiento.ipynb   ← código completo de esta etapa
├── docs/
├── .env                            ← (no está en GitHub, cada quien crea el suyo)
└── .gitignore
```

**El archivo que todos deben usar es:** `data/processed/tsla_processed.csv`

---

## 3. Features creadas (11 en total)

| # | Columna | Descripción |
|---|---|---|
| 1 | `open` | Precio de apertura del día |
| 2 | `high` | Precio más alto del día |
| 3 | `low` | Precio más bajo del día |
| 4 | `close` | Precio de cierre del día |
| 5 | `volume` | Volumen de acciones negociadas |
| 6 | `retorno_diario` | % de cambio respecto al día anterior (`pct_change`) |
| 7 | `sma_5` | Media móvil de 5 días del precio de cierre |
| 8 | `sma_20` | Media móvil de 20 días del precio de cierre |
| 9 | `volatilidad_5` | Desviación estándar de 5 días (medida de riesgo/inestabilidad) |
| 10 | `volumen_promedio_5` | Media móvil de 5 días del volumen |
| 11 | `precio_anterior` | Precio de cierre del día anterior (`shift(1)`) |

**Variable objetivo (target):**
- `precio_futuro` → precio de cierre del **día siguiente** (`shift(-1)`)

⚠️ Todas las features (columnas 1–11) están **escaladas** con `StandardScaler` de scikit-learn (media 0, desviación estándar 1). La columna `precio_futuro` se dejó **sin escalar**, en su valor real en dólares, porque es lo que se va a predecir.

---

## 4. Limpieza realizada

- Se eliminaron filas con valores nulos (`NaN`), generados por las ventanas móviles (`sma_20` necesita 20 días previos) y por el corrimiento (`shift`) al inicio/final del dataset.
- **Antes de limpiar:** 1,644 filas
- **Después de limpiar:** 1,624 filas
- Cumple de sobra el requisito de 800+ registros pedido por el profesor.

---

## 5. Cómo usar el dataset (para Persona 2, 3 y 4)

```python
import pandas as pd

df = pd.read_csv("../data/processed/tsla_processed.csv")

# Separar features (X) y variable objetivo (y)
X = df.drop(columns=["precio_futuro"])
y = df["precio_futuro"]

print(df.shape)   # (1624, 12)
df.head()
```

- **Persona 2 (EDA):** puede usar `df` directamente para gráficas, estadísticas descriptivas y matriz de correlación.
- **Persona 3 y 4 (Modelos):** ya tienen `X` (11 features escaladas) y `y` (precio futuro) listos para hacer el split `train_test_split` y entrenar sus modelos — **no hace falta volver a escalar ni limpiar nada.**

---

## 6. Pendiente / próximos pasos (equipo)

- [ ] Split train/test (sugerido: 80/20) — se puede definir en conjunto para que todos usen la misma división.
- [ ] EDA completo con gráficas (Persona 2)
- [ ] Entrenamiento de los 8 modelos (Persona 3 y 4)
- [ ] Tabla comparativa de métricas (MAE, MSE, RMSE, R²)
- [ ] Función de predicción con nuevo `.csv`
- [ ] Documento de infraestructura final (unir las 4 partes)

---

## 7. Notas técnicas para la exposición

- El uso de `StandardScaler` se justifica porque los modelos como SVM, KNN y redes neuronales (MLP) son sensibles a la escala de las variables — sin esto, columnas como `volume` (millones) dominarían sobre columnas como `retorno_diario` (decimales).
- El uso de medias móviles (`sma_5`, `sma_20`) y volatilidad son indicadores técnicos comunes en análisis financiero, y ayudan al modelo a capturar tendencias y riesgo, no solo el precio puntual de un día.
- La variable objetivo se definió como el precio del **día siguiente** (horizonte de predicción de 1 día) — esto debe explicarse claramente en la exposición ya que define qué tipo de predicción hace el sistema.
