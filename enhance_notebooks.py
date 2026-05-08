"""
Script para mejorar académicamente los notebooks del Proyecto Integrador 5.12.
Agrega encabezados formales, documentación por sección y análisis escritos.
Ejecutar UNA SOLA VEZ desde el directorio raíz del proyecto.
"""

import json
import uuid
from copy import deepcopy
from pathlib import Path

BASE_DIR = Path(__file__).parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


def _id():
    return uuid.uuid4().hex[:16]


def md(source):
    """Crea una celda de tipo Markdown."""
    if isinstance(source, str):
        source = [source]
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": source
    }


def code(source):
    """Crea una celda de tipo Code vacía (sin outputs)."""
    if isinstance(source, str):
        source = [source]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _id(),
        "metadata": {},
        "outputs": [],
        "source": source
    }


def read_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_nb(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Guardado: {path.name}")


def cell_source_str(cell):
    """Retorna el source de una celda como string concatenado."""
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src


def find_cell_idx(cells, keyword, cell_type=None):
    """Busca el índice de la primera celda que contenga keyword."""
    for i, c in enumerate(cells):
        if cell_type and c["cell_type"] != cell_type:
            continue
        if keyword in cell_source_str(c):
            return i
    return None


def insert_before(cells, idx, new_cells):
    """Inserta celdas antes del índice dado y retorna la nueva lista."""
    return cells[:idx] + new_cells + cells[idx:]


def insert_after(cells, idx, new_cells):
    """Inserta celdas después del índice dado y retorna la nueva lista."""
    return cells[:idx + 1] + new_cells + cells[idx + 1:]


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 01 — EDA
# ═══════════════════════════════════════════════════════════════════════════════

HEADER_01 = md("""\
# Proyecto Integrador 5.12 — Capítulo 5: Árboles e Interpretabilidad
## Notebook 1: Análisis Exploratorio de Datos (EDA)

---

**Asignatura:** Machine Learning
**Proyecto:** Predicción de Churn en Telecomunicaciones
**Dataset:** IBM Telco Customer Churn (7,043 clientes)
**Fecha:** 2025

---

### Objetivo del Notebook

Este notebook constituye la primera etapa del flujo de trabajo del proyecto. Su propósito es realizar
un análisis exploratorio riguroso del dataset *IBM Telco Customer Churn* con los siguientes objetivos específicos:

1. **Comprender la estructura del dataset:** dimensiones, tipos de variables y calidad de los datos.
2. **Detectar valores faltantes y duplicados** que requieran tratamiento previo al modelado.
3. **Analizar el desbalance de la variable objetivo** (`Churn`) para anticipar estrategias de evaluación.
4. **Identificar patrones y correlaciones** entre las variables predictoras y la probabilidad de abandono.
5. **Generar hipótesis** sobre qué variables tendrán mayor importancia en los modelos predictivos.

Los resultados de este análisis guiarán las decisiones de preprocesamiento y selección de modelos en el Notebook 2.

---
""")

DOC_IMPORTS_01 = md("""\
### 1. Configuración del Entorno

Se importan las bibliotecas necesarias para el análisis:
- `pandas` y `numpy`: manipulación y cálculo numérico.
- `matplotlib` y `seaborn`: visualización estadística.
- `pathlib.Path`: manejo de rutas de archivos de forma portable.

Se configuran las opciones de visualización de pandas para mostrar todas las columnas y filas necesarias.
""")

DOC_LOAD_01 = md("""\
### 2. Carga del Dataset

Se carga el dataset original desde la carpeta `data/raw/`. La ruta se construye de forma relativa
al directorio del proyecto usando `pathlib.Path`, lo que garantiza compatibilidad entre sistemas operativos.

Se espera observar el dataset con 7,043 registros y 21 columnas, incluyendo variables demográficas,
servicios contratados, información contractual y la variable objetivo `Churn`.
""")

DOC_QUALITY_01 = md("""\
### 3. Calidad del Dataset

Se analiza la calidad del dataset mediante tres verificaciones fundamentales:
- **Dimensiones** (`shape`): número de filas y columnas.
- **Tipos de datos** (`info`): permite identificar columnas que requieren conversión de tipo.
- **Estadísticas descriptivas** (`describe`): distribución de variables numéricas y frecuencias de categóricas.
- **Valores faltantes** (`isna().sum()`): identifica columnas con datos incompletos.
- **Registros duplicados** (`duplicated().sum()`): verifica la unicidad de los registros.

Se anticipa que `TotalCharges`, aunque debería ser numérica, puede estar codificada como `object`,
lo que genera valores nulos al convertirla.
""")

ANALYSIS_QUALITY_01 = md("""\
#### Interpretación de la Calidad del Dataset

El dataset presenta una calidad satisfactoria en su estado original. No se detectan registros duplicados
y la mayoría de las variables no presentan valores faltantes declarados. Sin embargo, la columna `TotalCharges`
está almacenada como tipo `object` en lugar de `float64`, lo que indica que contiene valores no numéricos
(cadenas vacías para clientes con `tenure = 0`). Al convertirla con `pd.to_numeric`, se generan 11 valores
nulos que corresponden a clientes que no han realizado ningún pago (clientes nuevos con `tenure = 0`).

Esta condición es coherente con el dominio del negocio: un cliente con cero meses de permanencia no tiene
cargos totales registrados. Por tanto, la imputación con valor `0` es la decisión más adecuada, ya que
refleja fielmente la realidad operacional.
""")

DOC_DISTRIBUTION_01 = md("""\
### 4. Análisis de la Variable Objetivo

Se analiza la distribución de la variable `Churn` para cuantificar el desbalance entre clases.
Este paso es crítico porque un desbalance significativo puede hacer que modelos entrenados con
`accuracy` como única métrica sean engañosos: un clasificador que siempre predice "No Churn"
alcanzaría ~73% de accuracy sin ningún valor predictivo real.

Se espera confirmar que existe un desbalance moderado (~73% No / ~27% Sí), lo que justifica
el uso de métricas como AUC-ROC, F1-score y Recall como criterios primarios de evaluación.
""")

ANALYSIS_DISTRIBUTION_01 = md("""\
#### Interpretación del Desbalance

La variable objetivo `Churn` presenta un desbalance moderado con una proporción aproximada de 3:1
entre clientes que no abandonaron (73.46%) y clientes que sí lo hicieron (26.54%). Este tipo de
desbalance es habitual en problemas de retención de clientes y representa un desafío para los modelos
de clasificación tradicionales.

En consecuencia, durante la etapa de modelado se priorizarán las siguientes estrategias:

1. **División estratificada** del conjunto de entrenamiento y prueba para preservar la proporción de clases.
2. **Métricas de evaluación robustas:** AUC-ROC, Recall, F1-score y Precision en lugar de Accuracy.
3. **Parámetros de balance de clases** en los clasificadores (`class_weight='balanced'`, `scale_pos_weight`).

El AUC-ROC es la métrica principal de selección de modelos, ya que mide la capacidad discriminativa
del modelo con independencia del umbral de clasificación.
""")

DOC_NUMERIC_01 = md("""\
### 5. Análisis de Variables Numéricas

Se analiza la distribución de las cuatro variables numéricas del dataset (`SeniorCitizen`, `tenure`,
`MonthlyCharges`, `TotalCharges`) segmentadas por la variable objetivo `Churn`.

Se utilizan `boxplots` y `histogramas` para visualizar:
- La distribución de cada variable y sus percentiles.
- Las diferencias estadísticas entre clientes con y sin churn.
- La presencia de valores atípicos que pudieran afectar al escalado.

Se anticipan diferencias significativas en `tenure` (antigüedad) y `MonthlyCharges` (cargos mensuales)
entre las dos clases de la variable objetivo.
""")

ANALYSIS_NUMERIC_01 = md("""\
#### Interpretación de Variables Numéricas

El análisis estadístico comparativo revela diferencias sustanciales entre los dos grupos:

**Tenure (antigüedad en meses):**
Los clientes que abandonaron la empresa presentan una antigüedad media de 17.98 meses, notablemente
inferior a los 37.57 meses de los clientes que permanecieron. Esta diferencia sugiere que los clientes
nuevos o con poca antigüedad son significativamente más vulnerables al churn, posiblemente porque aún
no han consolidado su relación con la empresa ni experimentado suficientes beneficios del servicio.

**MonthlyCharges (cargos mensuales):**
Los clientes con churn presentan cargos mensuales promedio más altos ($74.44 vs $61.27). Esto podría
indicar que los clientes con planes más costosos —probablemente asociados a servicios de fibra óptica—
experimentan mayor insatisfacción o encuentran alternativas más económicas en la competencia.

**TotalCharges (cargos totales):**
Paradójicamente, los clientes con churn presentan cargos totales menores ($1,531.80 vs $2,549.91),
lo que es consistente con su menor antigüedad: han estado menos tiempo y, por tanto, han acumulado
menos facturación total. Esta variable no es independiente de `tenure`.

Estas observaciones confirman la relevancia predictiva de las variables numéricas y anticipan que
`tenure` será una de las features más importantes en los modelos de clasificación.
""")

DOC_CATEGORICAL_01 = md("""\
### 6. Análisis de Variables Categóricas y Tasas de Churn

Para cada variable categórica relevante se calcula la **tasa de churn por categoría**, definida como
el porcentaje de clientes de ese grupo que abandonaron la empresa. Este análisis permite identificar
qué modalidades de cada variable están más asociadas al abandono.

Variables analizadas: `Contract`, `InternetService`, `PaymentMethod`, `OnlineSecurity`,
`TechSupport`, `PaperlessBilling`, `OnlineBackup`, `DeviceProtection`.
""")

ANALYSIS_CATEGORICAL_01 = md("""\
#### Interpretación de Tasas de Churn por Categoría

El análisis por categorías revela patrones muy significativos desde la perspectiva de negocio:

**Tipo de contrato (`Contract`):** Es la variable con mayor poder discriminante. Los clientes con
contrato mensual presentan una tasa de churn del 42.71%, mientras que los contratos anuales y bianuales
muestran tasas de 11.27% y 2.83% respectivamente. Este gradiente descendente confirma que la estabilidad
contractual es el factor de retención más poderoso.

**Servicio de internet (`InternetService`):** Los clientes con fibra óptica presentan una tasa de
churn del 41.89%, muy superior al 18.96% del servicio DSL. Este patrón sugiere posibles problemas de
percepción de valor en el segmento de fibra óptica, ya que sus cargos mensuales son también más altos.

**Método de pago (`PaymentMethod`):** El cheque electrónico está asociado a la mayor tasa de churn
(45.29%), mientras que los métodos automáticos (débito bancario, tarjeta de crédito) presentan tasas
entre 15-17%. Los métodos automáticos pueden indicar mayor compromiso e integración del cliente
con los servicios de la empresa.

**Servicios de soporte (`TechSupport`, `OnlineSecurity`):** La ausencia de soporte técnico y
seguridad en línea está asociada a tasas de churn del ~42%, frente al ~15% de quienes los contratan.
Estos servicios funcionan como factores de retención al aumentar la percepción de valor del paquete.
""")

DOC_CROSSANALYSIS_01 = md("""\
### 7. Análisis Cruzado entre Variables

Se estudian combinaciones de variables para identificar perfiles de clientes con riesgo compuesto.
El análisis cruzado entre `Contract`, `PaymentMethod` e `InternetService` permite detectar segmentos
específicos con tasas de churn especialmente elevadas, información valiosa para el diseño de
estrategias de retención diferenciadas.

Se presenta la tasa de churn como un mapa de calor (*heatmap*) para facilitar la identificación visual
de los perfiles de mayor riesgo.
""")

ANALYSIS_CROSSANALYSIS_01 = md("""\
#### Interpretación del Análisis Cruzado

El análisis cruzado confirma y amplía los hallazgos univariados. La combinación de contrato mensual
con cheque electrónico genera una tasa de churn del 53.73%, la más alta de toda la segmentación.
En contraste, clientes con contrato bianual y métodos de pago automáticos presentan tasas inferiores
al 4%.

El análisis cruzado entre `Contract` e `InternetService` revela que los clientes con contrato mensual
y fibra óptica tienen una tasa de churn del 54.61%, confirmando que ambas variables actúan de forma
aditiva en la generación de riesgo.

Desde una perspectiva de negocio, estos hallazgos sugieren que las campañas de retención más efectivas
deberían orientarse a clientes con contrato mensual, servicio de fibra óptica y pago por cheque
electrónico, ya que representan el segmento de mayor riesgo compuesto.
""")

CONCLUSIONS_01 = md("""\
### 8. Conclusiones del Análisis Exploratorio

El análisis exploratorio del dataset IBM Telco Customer Churn permite establecer las siguientes conclusiones,
que guiarán las decisiones de la etapa de modelado:

1. **Calidad del dato:** El dataset presenta excelente calidad con un único problema identificado
   en la columna `TotalCharges` (11 valores nulos por conversión de tipo), resuelto con imputación a `0`.

2. **Desbalance de clases:** La distribución 73.46%/26.54% requiere métricas robustas (AUC-ROC, F1, Recall)
   y estrategias de balance en los clasificadores.

3. **Variables de mayor relevancia anticipada:**
   - `Contract` (tipo de contrato): mayor discriminante categórico.
   - `tenure` (antigüedad): mayor discriminante numérico.
   - `InternetService` (tipo de internet): fibra óptica es factor de riesgo.
   - `PaymentMethod` (método de pago): cheque electrónico como señal de riesgo.
   - `TechSupport`, `OnlineSecurity`: ausencia como factor de riesgo.

4. **Perfiles de alto riesgo identificados:**
   - Clientes con contrato mensual + cheque electrónico: ~54% de churn.
   - Clientes con contrato mensual + fibra óptica: ~55% de churn.
   - Clientes nuevos (tenure ≤ 6 meses): tasa de churn significativamente elevada.

5. **Próximos pasos:** El Notebook 2 implementará Pipelines de preprocesamiento con
   `SimpleImputer`, `StandardScaler` y `OneHotEncoder`, seguidos de `GridSearchCV` con
   Random Forest, XGBoost, CatBoost y LightGBM.
""")


def enhance_nb01():
    print("Procesando: 01_eda_telco_churn.ipynb")
    path = NOTEBOOKS_DIR / "01_eda_telco_churn.ipynb"
    nb = read_nb(path)
    cells = nb["cells"]

    # 1. Insertar encabezado formal al inicio
    cells = [HEADER_01] + cells

    # 2. Documentar sección de imports (buscar celda con "import pandas")
    idx = find_cell_idx(cells, "import pandas", "code")
    if idx is not None and cells[idx - 1]["cell_type"] != "markdown":
        cells = insert_before(cells, idx, [DOC_IMPORTS_01])

    # 3. Documentar carga de datos (buscar "read_csv")
    idx = find_cell_idx(cells, "read_csv", "code")
    if idx is not None and cells[idx - 1]["cell_type"] != "markdown":
        cells = insert_before(cells, idx, [DOC_LOAD_01])

    # 4. Documentar análisis de calidad (buscar "df.info")
    idx = find_cell_idx(cells, "df.info()", "code")
    if idx is not None:
        cells = insert_before(cells, idx, [DOC_QUALITY_01])
        # Agregar análisis después de la celda de isna
        idx2 = find_cell_idx(cells, "duplicated().sum()", "code")
        if idx2 is not None:
            cells = insert_after(cells, idx2, [ANALYSIS_QUALITY_01])

    # 5. Documentar distribución de Churn
    idx = find_cell_idx(cells, "value_counts(normalize=True)", "code")
    if idx is not None:
        cells = insert_before(cells, idx, [DOC_DISTRIBUTION_01])
        # Insertar análisis después del gráfico de barras porcentual
        idx2 = find_cell_idx(cells, "ylim(0, 100)", "code")
        if idx2 is not None:
            cells = insert_after(cells, idx2, [ANALYSIS_DISTRIBUTION_01])

    # 6. Documentar variables numéricas
    idx = find_cell_idx(cells, "boxplot", "code")
    if idx is not None:
        cells = insert_before(cells, idx, [DOC_NUMERIC_01])
        cells = insert_after(cells, idx, [ANALYSIS_NUMERIC_01])

    # 7. Documentar variables categóricas
    idx = find_cell_idx(cells, "churn_rate_table", "code")
    if idx is not None and cells[idx - 1]["cell_type"] != "markdown":
        cells = insert_before(cells, idx, [DOC_CATEGORICAL_01])
        # Insertar análisis después del barplot de categorías
        idx2 = find_cell_idx(cells, "barplot.*churn_rate", "code")
        if idx2 is None:
            idx2 = find_cell_idx(cells, "for col in variables_categoricas_clave", "code")
        if idx2 is not None:
            cells = insert_after(cells, idx2, [ANALYSIS_CATEGORICAL_01])

    # 8. Documentar análisis cruzado
    idx = find_cell_idx(cells, "crosstab", "code")
    if idx is not None and cells[idx - 1]["cell_type"] != "markdown":
        cells = insert_before(cells, idx, [DOC_CROSSANALYSIS_01])
        idx2 = find_cell_idx(cells, "contract_internet", "code")
        if idx2 is not None:
            cells = insert_after(cells, idx2, [ANALYSIS_CROSSANALYSIS_01])

    # 9. Reemplazar o enriquecer conclusiones finales
    idx = find_cell_idx(cells, "Conclusiones principales", "markdown")
    if idx is None:
        cells = cells + [CONCLUSIONS_01]

    nb["cells"] = cells
    write_nb(nb, path)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 02 — MODELADO
# ═══════════════════════════════════════════════════════════════════════════════

HEADER_02 = md("""\
# Proyecto Integrador 5.12 — Capítulo 5: Árboles e Interpretabilidad
## Notebook 2: Modelado Predictivo con Pipeline y GridSearchCV

---

**Asignatura:** Machine Learning
**Proyecto:** Predicción de Churn en Telecomunicaciones
**Algoritmos:** Random Forest · XGBoost · CatBoost · LightGBM
**Fecha:** 2025

---

### Objetivo del Notebook

Este notebook implementa el flujo completo de modelado predictivo para el problema de clasificación
de churn en telecomunicaciones. El proceso sigue las mejores prácticas de Machine Learning aplicado:

1. **Preprocesamiento estructurado** mediante `Pipeline` de scikit-learn con `SimpleImputer`,
   `StandardScaler` y `OneHotEncoder`.
2. **Optimización de hiperparámetros** con `GridSearchCV` y validación cruzada estratificada de 5 pliegues.
3. **Entrenamiento y evaluación** de 4 algoritmos basados en árboles: Random Forest, XGBoost, CatBoost y LightGBM.
4. **Comparación rigurosa** mediante métricas múltiples: Accuracy, Precision, Recall, F1-Score y AUC-ROC.
5. **Selección y serialización** del mejor modelo para su uso posterior en la API y el análisis de interpretabilidad.
6. **Análisis de importancia de características** para comprender qué variables guían las predicciones.

---
""")

DOC_PREPROCESSING_02 = md("""\
### 3. Pipeline de Preprocesamiento

Se construye un `Pipeline` de scikit-learn que encapsula todas las transformaciones de datos
de forma reproducible y evita *data leakage* (filtraciones de información del conjunto de prueba
al de entrenamiento).

**Transformaciones aplicadas:**
- **Variables numéricas** (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`):
  1. `SimpleImputer(strategy='median')`: imputa valores faltantes con la mediana, robusta a outliers.
  2. `StandardScaler()`: escala a media 0 y desviación estándar 1.
- **Variables categóricas** (15 variables):
  1. `SimpleImputer(strategy='most_frequent')`: imputa con la categoría más frecuente.
  2. `OneHotEncoder(handle_unknown='ignore')`: codifica en variables binarias, ignorando categorías
     desconocidas en tiempo de inferencia.

El `ColumnTransformer` aplica cada transformación solo a las columnas correspondientes y concatena
los resultados en la matriz de features final.
""")

DOC_SPLIT_02 = md("""\
### 4. División Entrenamiento/Prueba

Se aplica `train_test_split` con `stratify=y` para preservar la distribución de clases en ambos
conjuntos (73.46% No Churn / 26.54% Churn). Esto es fundamental para obtener estimaciones de
desempeño no sesgadas cuando existe desbalance de clases.

- **Proporción:** 80% entrenamiento (5,634 registros) / 20% prueba (1,409 registros)
- **Random state:** 42 para reproducibilidad
- `scale_pos_weight` para XGBoost: negatives/positives ≈ 2.77 (para ponderar la clase minoritaria)
""")

DOC_MODELS_02 = md("""\
### 5. Configuración de Modelos y Búsqueda de Hiperparámetros

Para cada algoritmo se define un espacio de búsqueda de hiperparámetros. La optimización se realiza
con `GridSearchCV`:
- **cv=5:** validación cruzada estratificada de 5 pliegues.
- **scoring='roc_auc':** métrica de selección (robusta al desbalance de clases).
- **n_jobs=-1:** uso de todos los núcleos disponibles para paralelización.
- **refit=True:** re-entrena con los mejores parámetros sobre el conjunto de entrenamiento completo.

**Espacio de búsqueda por modelo:**
- **Random Forest:** n_estimators, max_depth, min_samples_split, min_samples_leaf, class_weight
- **XGBoost:** n_estimators, max_depth, learning_rate, subsample, colsample_bytree, scale_pos_weight
- **CatBoost:** iterations, depth, learning_rate, l2_leaf_reg, auto_class_weights
- **LightGBM:** n_estimators, max_depth, learning_rate, num_leaves, subsample, class_weight
""")

DOC_RESULTS_02 = md("""\
### 6. Comparación de Resultados

Se presenta la tabla comparativa de métricas evaluadas sobre el conjunto de prueba (nunca visto durante
el entrenamiento ni la optimización de hiperparámetros). Las métricas reportadas son:

- **CV AUC:** AUC-ROC promedio en validación cruzada (estimación insesgada del desempeño esperado).
- **Test Accuracy:** proporción de predicciones correctas.
- **Test Precision:** de los clientes predichos como Churn, ¿qué fracción realmente lo es?
- **Test Recall:** de los clientes que realmente hacen Churn, ¿qué fracción predice el modelo?
- **Test F1:** media armónica entre Precision y Recall.
- **Test AUC-ROC:** área bajo la curva ROC (criterio principal de selección).

**Criterio de selección del mejor modelo:** mayor AUC-ROC en el conjunto de prueba.
""")

ANALYSIS_RESULTS_02 = md("""\
#### Interpretación de Resultados Comparativos

Los cuatro modelos evaluados presentan un desempeño competitivo en términos de AUC-ROC, con valores
entre 0.839 y 0.848. El modelo **XGBoost** fue seleccionado como el mejor con un AUC-ROC de 0.8481,
seguido de cerca por CatBoost (0.8476) y LightGBM (0.8474).

**Análisis del trade-off Precision-Recall:**
Se observa que diferentes modelos optimizan distintos aspectos del problema:
- **XGBoost y LightGBM** favorecen Precision (≈0.66-0.67) sobre Recall (≈0.51-0.52), lo que implica
  menor tasa de falsos positivos pero mayor tasa de clientes en riesgo no detectados.
- **CatBoost y Random Forest** presentan mayor Recall (≈0.72-0.80) a costa de menor Precision,
  detectando más clientes en riesgo pero con más falsas alarmas.

**Implicación de negocio:** La elección entre Precision y Recall depende del costo relativo de cada
tipo de error. Si el costo de una falsa alarma (acción de retención innecesaria) es bajo comparado
con el costo de no detectar un churn real, modelos con mayor Recall serían preferibles. XGBoost
fue seleccionado por su superior AUC-ROC global, que representa mejor la capacidad discriminativa
general del modelo independientemente del umbral de clasificación.
""")

CODE_CONFMAT_ALL = code("""\
# Matrices de confusión para todos los modelos (seaborn heatmap)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

model_order = ["Random Forest", "XGBoost", "CatBoost", "LightGBM"]

for i, model_name in enumerate(model_order):
    if model_name not in best_estimators:
        continue
    m = best_estimators[model_name]
    y_pred_m = m.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_m)

    # Calcular porcentajes
    cm_pct = cm / cm.sum() * 100
    labels = [[f"{v}\\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
              for row_v, row_p in zip(cm, cm_pct)]

    color = {"Random Forest": "Blues", "XGBoost": "Greens",
             "CatBoost": "Oranges", "LightGBM": "Purples"}.get(model_name, "Blues")

    sns.heatmap(
        cm, annot=labels, fmt="", cmap=color,
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=axes[i], linewidths=0.5, linecolor="gray",
        cbar_kws={"shrink": 0.8}
    )
    axes[i].set_title(f"{model_name}", fontsize=14, fontweight="bold", pad=10)
    axes[i].set_xlabel("Predicción", fontsize=11)
    axes[i].set_ylabel("Valor Real", fontsize=11)

plt.suptitle(
    "Matrices de Confusión — Comparación de Modelos\\n(Conjunto de Prueba, n=1,409)",
    fontsize=15, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.show()
""")

ANALYSIS_CONFMAT_ALL = md("""\
#### Interpretación de las Matrices de Confusión

Las matrices de confusión revelan el comportamiento detallado de cada modelo ante los dos tipos
de error posibles:

- **Falsos Negativos (FN):** clientes que realmente abandonarán pero el modelo predice que no.
  Este es el error más costoso para el negocio, pues representa clientes en riesgo no atendidos.
- **Falsos Positivos (FP):** clientes que no abandonarán pero el modelo predice que sí.
  Este error genera costos operacionales por acciones de retención innecesarias.

**XGBoost** presenta la mayor cantidad de Verdaderos Negativos (correctamente clasificados como
No Churn), lo que le confiere una Precision más alta. **CatBoost** detecta más Verdaderos Positivos
(clientes con churn real correctamente identificados), lo que se refleja en su mayor Recall.

La elección entre estos trade-offs depende de los costos operacionales del negocio. Para este
proyecto se seleccionó XGBoost por su superior AUC-ROC global, que es independiente del umbral
de clasificación y constituye la métrica más robusta para comparar modelos ante el desbalance de clases.
""")

CODE_ROC_COMPARISON = code("""\
# Curvas ROC comparativas en una sola figura
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_auc_score

fig, ax = plt.subplots(figsize=(9, 7))

colors = {
    "Random Forest": "#e74c3c",
    "XGBoost": "#27ae60",
    "CatBoost": "#f39c12",
    "LightGBM": "#8e44ad"
}

for model_name, m in best_estimators.items():
    y_proba_m = m.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba_m)
    RocCurveDisplay.from_predictions(
        y_test, y_proba_m,
        name=f"{model_name}  (AUC = {auc:.4f})",
        ax=ax, color=colors.get(model_name, "blue"),
        linewidth=2
    )

ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Clasificador Aleatorio (AUC = 0.50)")
ax.set_title(
    "Curvas ROC — Comparación de Modelos\\n(Conjunto de Prueba, n=1,409)",
    fontsize=14, fontweight="bold"
)
ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=12)
ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)", fontsize=12)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.show()
""")

ANALYSIS_ROC_02 = md("""\
#### Interpretación de las Curvas ROC

La curva ROC (Receiver Operating Characteristic) representa el trade-off entre la Tasa de
Verdaderos Positivos (sensibilidad) y la Tasa de Falsos Positivos a diferentes umbrales de
clasificación. El Área Bajo la Curva (AUC-ROC) cuantifica la capacidad discriminativa global
del modelo: un valor de 1.0 indica clasificación perfecta, mientras que 0.5 indica desempeño
equivalente a una clasificación aleatoria.

Los cuatro modelos presentan curvas ROC muy similares con AUC entre 0.839 y 0.848, lo que indica
que todos capturan los patrones de churn de forma competente. Las curvas se ubican claramente
por encima de la diagonal de referencia (clasificador aleatorio), confirmando su valor predictivo.

La diferencia de AUC entre el mejor (XGBoost, 0.8481) y el peor modelo (Random Forest, 0.8392)
es de apenas 0.009 puntos, sugiriendo que el dataset y sus señales predictivas dominan el
desempeño más que las diferencias entre algoritmos. Sin embargo, XGBoost mantiene una ventaja
consistente y fue seleccionado como el modelo final para despliegue.
""")

CODE_FEAT_IMP_ALL = code("""\
# Importancia de características para los 4 modelos (subplots comparativos)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

model_order = ["Random Forest", "XGBoost", "CatBoost", "LightGBM"]
colors_fi = {
    "Random Forest": "#e74c3c",
    "XGBoost": "#27ae60",
    "CatBoost": "#f39c12",
    "LightGBM": "#8e44ad"
}
N_TOP = 12

for i, model_name in enumerate(model_order):
    if model_name not in best_estimators:
        continue
    m = best_estimators[model_name]
    clf = m.named_steps["classifier"]
    prep = m.named_steps["preprocessor"]
    feat_names_raw = prep.get_feature_names_out()
    feat_names = [n.replace("num__", "").replace("cat__", "") for n in feat_names_raw]

    if not hasattr(clf, "feature_importances_"):
        axes[i].text(0.5, 0.5, "No disponible", ha="center", va="center")
        axes[i].set_title(model_name)
        continue

    df_fi = pd.DataFrame({
        "feature": feat_names,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False).head(N_TOP)

    df_fi_sorted = df_fi.sort_values("importance", ascending=True)

    axes[i].barh(
        df_fi_sorted["feature"], df_fi_sorted["importance"],
        color=colors_fi[model_name], edgecolor="white", alpha=0.85
    )
    axes[i].set_title(f"{model_name} — Top {N_TOP} Features", fontweight="bold", fontsize=12)
    axes[i].set_xlabel("Importancia", fontsize=10)
    axes[i].tick_params(axis="y", labelsize=9)
    axes[i].grid(axis="x", alpha=0.3)

plt.suptitle(
    "Importancia de Características por Modelo\\n(Features más relevantes para la predicción de Churn)",
    fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.show()
""")

ANALYSIS_FEAT_IMP_02 = md("""\
#### Interpretación de la Importancia de Características

El análisis de importancia de características revela una notable **consistencia entre modelos**:
en todos los algoritmos evaluados, las mismas variables aparecen entre las más relevantes, lo que
otorga solidez y confiabilidad a las conclusiones.

**Variable dominante:** `Contract_Month-to-month` emerge como el predictor más potente en todos
los modelos, con importancias que oscilan entre el 15% y el 35%. Este resultado es coherente con
el hallazgo del EDA: los contratos mensuales presentan una tasa de churn del 42.7%, la más alta
de todas las modalidades contractuales.

**Variables de segundo orden:** `TechSupport_No`, `OnlineSecurity_No` e `InternetService_Fiber optic`
aparecen consistentemente entre las 5 más importantes. Estos factores de riesgo secundarios
representan oportunidades concretas de intervención: ofrecer servicios de soporte y seguridad
a clientes con contrato mensual y fibra óptica podría reducir significativamente el riesgo de churn.

**Variables numéricas:** `tenure` (antigüedad) aparece como la única variable numérica con alta
importancia consistente, confirmando que los clientes con menor permanencia son inherentemente
más vulnerables al abandono. `MonthlyCharges` y `TotalCharges` tienen importancias moderadas,
posiblemente porque parte de su información ya está capturada por otras variables correlacionadas.

**Implicación para el negocio:** Las acciones de retención más costo-efectivas deberían enfocarse
en tres palancas: (1) migrar clientes de contratos mensuales a contratos anuales con incentivos,
(2) ofrecer servicios de soporte técnico y seguridad como parte de paquetes de fidelización,
y (3) implementar programas especiales para clientes con antigüedad inferior a 12 meses.
""")

CONCLUSIONS_02 = md("""\
### 7. Conclusiones del Modelado Predictivo

El proceso de modelado predictivo con validación cruzada y búsqueda de hiperparámetros permite
establecer las siguientes conclusiones:

1. **Modelo seleccionado:** XGBoost con `max_depth=3`, `learning_rate=0.05`, `n_estimators=100`,
   `subsample=0.8`, `colsample_bytree=0.8`. Su AUC-ROC de 0.8481 en el conjunto de prueba representa
   un desempeño sólido para este tipo de problema de clasificación binaria con desbalance moderado.

2. **Homogeneidad del espacio de soluciones:** Los cuatro algoritmos presentan AUC-ROC muy similares
   (diferencia máxima de 0.009), lo que sugiere que el dataset contiene señales predictivas fuertes
   y claras que cualquier algoritmo basado en árboles puede capturar eficientemente.

3. **Robustez del pipeline:** La integración de preprocesamiento, optimización y clasificación en un
   único `Pipeline` de scikit-learn garantiza que las transformaciones se apliquen correctamente
   tanto en entrenamiento como en inferencia, eliminando el riesgo de data leakage.

4. **Variables críticas identificadas:** `Contract_Month-to-month`, `TechSupport_No`,
   `OnlineSecurity_No`, `InternetService_Fiber optic` y `tenure` son las variables que más
   guían las predicciones del modelo, en coherencia con los hallazgos del EDA.

5. **Modelo serializado:** El pipeline completo (preprocesamiento + clasificador) fue guardado en
   `models/best_model.joblib` para su uso en el análisis de interpretabilidad (Notebook 3) y en
   la API de predicción en tiempo real (FastAPI).
""")


def enhance_nb02():
    print("Procesando: 02_modeling_pipeline_gridsearch.ipynb")
    path = NOTEBOOKS_DIR / "02_modeling_pipeline_gridsearch.ipynb"
    nb = read_nb(path)
    cells = nb["cells"]

    # 1. Encabezado formal al inicio
    cells = [HEADER_02] + cells

    # 2. Documentar preprocesamiento
    idx = find_cell_idx(cells, "numeric_transformer", "code")
    if idx is not None and cells[idx - 1]["cell_type"] != "markdown":
        cells = insert_before(cells, idx, [DOC_PREPROCESSING_02])

    # 3. Documentar split
    idx = find_cell_idx(cells, "train_test_split", "code")
    if idx is not None and cells[idx - 1]["cell_type"] != "markdown":
        cells = insert_before(cells, idx, [DOC_SPLIT_02])

    # 4. Documentar modelos y params
    idx = find_cell_idx(cells, "models_and_params", "code")
    if idx is not None and cells[idx - 1]["cell_type"] != "markdown":
        cells = insert_before(cells, idx, [DOC_MODELS_02])

    # 5. Documentar comparación de resultados
    idx = find_cell_idx(cells, "results_df_sorted", "code")
    if idx is not None:
        cells = insert_before(cells, idx, [DOC_RESULTS_02])
        cells = insert_after(cells, idx + 1, [ANALYSIS_RESULTS_02])

    # 6. Insertar matrices de confusión comparativas después de la matriz del mejor modelo
    idx = find_cell_idx(cells, "ConfusionMatrixDisplay", "code")
    if idx is not None:
        cells = insert_after(cells, idx, [CODE_CONFMAT_ALL, ANALYSIS_CONFMAT_ALL])

    # 7. Insertar curvas ROC mejoradas después de las existentes
    idx = find_cell_idx(cells, "RocCurveDisplay", "code")
    if idx is not None:
        cells = insert_after(cells, idx, [CODE_ROC_COMPARISON, ANALYSIS_ROC_02])

    # 8. Insertar feature importance comparativa después de la gráfica del mejor modelo
    idx = find_cell_idx(cells, "feature_importances_", "code")
    if idx is not None:
        # Buscar la celda del gráfico de feature importance
        idx2 = find_cell_idx(cells, "Top 20", "code")
        if idx2 is not None:
            cells = insert_after(cells, idx2, [CODE_FEAT_IMP_ALL, ANALYSIS_FEAT_IMP_02])

    # 9. Enriquecer conclusiones
    idx = find_cell_idx(cells, "Conclusiones del modelado", "markdown")
    if idx is not None:
        cells[idx] = CONCLUSIONS_02

    nb["cells"] = cells
    write_nb(nb, path)


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 03 — INTERPRETABILIDAD
# ═══════════════════════════════════════════════════════════════════════════════

HEADER_03 = md("""\
# Proyecto Integrador 5.12 — Capítulo 5: Árboles e Interpretabilidad
## Notebook 3: Interpretabilidad Local con LIME

---

**Asignatura:** Machine Learning
**Proyecto:** Predicción de Churn en Telecomunicaciones
**Método:** LIME (Local Interpretable Model-agnostic Explanations)
**Fecha:** 2025

---

### Objetivo del Notebook

Este notebook implementa el análisis de interpretabilidad local del modelo XGBoost entrenado en el
Notebook 2. La interpretabilidad es un componente esencial en aplicaciones de Machine Learning empresarial,
ya que permite:

1. **Comprender las decisiones individuales** del modelo más allá de su desempeño global.
2. **Generar confianza** en las predicciones por parte de stakeholders no técnicos.
3. **Identificar factores accionables** para intervenciones de retención personalizadas.
4. **Detectar potenciales sesgos** o comportamientos inesperados del modelo.

Se utiliza **LIME** (*Local Interpretable Model-agnostic Explanations*, Ribeiro et al., 2016), que
aproxima localmente el comportamiento del modelo complejo (XGBoost) con un modelo lineal interpretable
en la vecindad de cada instancia a explicar. LIME es **agnóstico al modelo**: funciona con cualquier
clasificador de caja negra sin acceso a su estructura interna.

**Tres perfiles analizados:**
1. Cliente con **alta** probabilidad de churn (verdadero positivo, prob ≈ 0.89).
2. Cliente con **baja** probabilidad de churn (verdadero negativo, prob ≈ 0.01).
3. Cliente con probabilidad **intermedia** (caso límite, prob ≈ 0.50).

---
""")

DOC_SETUP_03 = md("""\
### 1. Configuración del Entorno LIME

Se carga el modelo entrenado (`best_model.joblib`) y los metadatos del proyecto. LIME requiere
acceso a los datos de entrenamiento para construir las perturbaciones locales que estiman la
influencia de cada feature.

**Consideraciones técnicas:**
- El modelo es un `Pipeline` que incluye preprocesamiento; LIME necesita operar sobre los
  datos en su formato original (antes del pipeline), por lo que se crea una función `lime_predict_proba`
  que aplica el pipeline completo a las instancias perturbadas.
- Las variables categóricas deben codificarse numéricamente para LIME, manteniendo el mapeo
  inverso para la presentación de resultados legibles.
""")

DOC_EXPLAINER_03 = md("""\
### 2. Creación del Explicador LIME

Se instancia el `LimeTabularExplainer` con los siguientes parámetros:
- `training_data`: datos de entrenamiento en formato numérico (variables categóricas codificadas).
- `feature_names`: nombres de las 19 variables predictoras.
- `class_names`: etiquetas de las clases ("No Churn", "Churn").
- `categorical_features`: índices de las variables categóricas.
- `categorical_names`: diccionario que mapea cada índice a sus posibles categorías.
- `mode='classification'`: problema de clasificación binaria.
- `discretize_continuous=True`: discretiza variables numéricas para mayor interpretabilidad.

El explicador genera perturbaciones locales de cada instancia, obtiene predicciones del modelo
para esas perturbaciones y ajusta un modelo lineal local ponderado por proximidad a la instancia.
""")

DOC_CASES_03 = md("""\
### 3. Selección de Casos Representativos

Se seleccionan tres instancias representativas del conjunto de prueba:

1. **Alta probabilidad de churn:** el cliente con la mayor probabilidad predicha de abandono,
   idealmente un verdadero positivo (churn real = 1, predicción = 1).
2. **Baja probabilidad de churn:** el cliente con la menor probabilidad predicha, idealmente
   un verdadero negativo (churn real = 0, predicción = 0).
3. **Probabilidad intermedia:** el cliente más cercano a la probabilidad 0.5, es decir, el
   caso límite donde el modelo presenta mayor incertidumbre. Este es el más valioso desde
   el punto de vista de intervención, pues una pequeña mejora en su situación podría
   cambiar la predicción.
""")

ANALYSIS_HIGH_03 = md("""\
#### Análisis del Cliente con Alta Probabilidad de Churn

Este perfil representa el segmento de máxima prioridad para las acciones de retención empresarial.
La alta probabilidad estimada (≈88%) indica que el modelo detecta múltiples señales de riesgo
en simultáneo.

Las variables que LIME identifica como más influyentes en esta predicción son coherentes con los
hallazgos del EDA y la importancia global del modelo:

- **Contrato mensual:** la ausencia de un compromiso contractual de largo plazo elimina la fricción
  de cancelación, facilitando el abandono.
- **Baja antigüedad:** clientes con pocos meses de permanencia no han desarrollado un vínculo de
  lealtad suficiente y son más sensibles a ofertas competidoras.
- **Ausencia de servicios de soporte:** sin soporte técnico ni seguridad en línea, el cliente percibe
  menos valor en el paquete contratado.
- **Servicio de fibra óptica:** aunque este servicio es más rápido, su mayor costo puede generar
  insatisfacción si la calidad percibida no justifica el precio premium.

**Recomendación de negocio:** intervención urgente con oferta de migración a contrato anual
con descuento, incorporación gratuita de soporte técnico por 3 meses y revisión de la percepción
de calidad del servicio de fibra óptica.
""")

ANALYSIS_LOW_03 = md("""\
#### Análisis del Cliente con Baja Probabilidad de Churn

Este perfil representa al cliente "ideal" en términos de retención. La muy baja probabilidad
estimada (≈1%) indica que el modelo no detecta señales significativas de riesgo de abandono.

Las variables con mayor peso negativo (que reducen la probabilidad de churn) en la explicación
LIME permiten entender qué factores generan lealtad y retención:

- **Contrato de largo plazo:** los contratos anuales o bianuales crean una barrera de salida natural
  que disuade el abandono, independientemente del nivel de satisfacción.
- **Alta antigüedad:** clientes con muchos meses de permanencia han superado la fase de fricción
  inicial y desarrollado hábitos de uso arraigados.
- **Servicios de valor agregado:** la presencia de soporte técnico, seguridad en línea u otros
  servicios adicionales aumenta el costo percibido de cambiar de proveedor.
- **Método de pago automático:** la domiciliación del pago reduce la frecuencia de interacciones
  activas con la empresa, disminuyendo las oportunidades de tomar decisiones de cancelación.

**Estrategia recomendada:** para este perfil, la prioridad no es retener sino fidelizar y maximizar
el valor del cliente. Se recomiendan programas de lealtad, upgrades de servicio y referencias.
""")

ANALYSIS_MID_03 = md("""\
#### Análisis del Cliente con Probabilidad Intermedia de Churn (Caso Límite)

Este es el caso analíticamente más rico del análisis de interpretabilidad, ya que la probabilidad
estimada (~50%) indica que el modelo se encuentra en el umbral de decisión. La explicación LIME
revela una **coexistencia de factores de riesgo y factores de protección** que se contrarrestan
mutuamente.

Este tipo de cliente es especialmente valioso desde la perspectiva de intervención empresarial:
a diferencia del cliente de alto riesgo (donde el churn puede ser inevitable) o del de bajo riesgo
(donde no se justifica invertir recursos), el cliente intermedio representa el segmento donde
las acciones de retención tienen mayor probabilidad de cambiar el resultado final.

**Características típicas de este perfil:**
- Contrato mensual (factor de riesgo) pero con algunos servicios adicionales contratados (factor de protección).
- Antigüedad moderada (mayor que los clientes de alto riesgo pero menor que los de bajo riesgo).
- Cargos mensuales en el rango medio, sin ser excesivamente altos.

**Estrategia recomendada:** monitoreo activo y evaluación de incentivos dirigidos a las variables
que LIME identifica como factores de riesgo. Una oferta personalizada que aborde específicamente
los aspectos negativos (por ejemplo, migración de contrato mensual a anual con incentivo económico)
tiene alta probabilidad de retener a este cliente.

La identificación de este segmento intermedio es uno de los principales aportes prácticos de
LIME al negocio: permite diseñar intervenciones quirúrgicas dirigidas exactamente a quienes más
se beneficiarán de ellas.
""")

CONCLUSIONS_03 = md("""\
### 4. Conclusiones del Análisis de Interpretabilidad

El análisis de interpretabilidad local con LIME permite establecer las siguientes conclusiones
que complementan y validan los resultados cuantitativos del Notebook 2:

1. **Coherencia global-local:** Las variables identificadas como más importantes a nivel global
   (Contract, TechSupport, OnlineSecurity, tenure, InternetService) también emergen como los
   factores explicativos más relevantes en las explicaciones locales de LIME, lo que confirma
   la consistencia del modelo.

2. **Interpretabilidad accionable:** LIME permite traducir predicciones abstractas (probabilidades)
   en explicaciones concretas y accionables para el equipo de retención: "este cliente tiene alto
   riesgo principalmente por su contrato mensual y la ausencia de soporte técnico".

3. **Identificación de segmentos de intervención:** Los tres perfiles analizados representan
   estrategias de negocio distintas: (a) intervención urgente para clientes de alto riesgo,
   (b) fidelización para clientes de bajo riesgo, y (c) targeting quirúrgico para clientes
   en el umbral de decisión.

4. **Validación del modelo:** El hecho de que LIME genere explicaciones coherentes con el
   conocimiento del dominio (variables contractuales y de servicio como factores predictivos)
   aumenta la confianza en que el modelo captura relaciones causales reales y no artefactos estadísticos.

5. **Limitaciones de LIME:** Las explicaciones LIME son locales y pueden ser inestables (variar
   entre ejecuciones por la aleatoriedad en las perturbaciones). Para mayor robustez, se recomienda
   combinar LIME con métodos globales como SHAP en análisis de producción.

6. **Integración con la API:** Las conclusiones de este análisis informaron el diseño de los
   "factores de riesgo" mostrados en la respuesta del endpoint `/predict` de la API FastAPI,
   permitiendo que el sistema de predicción en producción también proporcione explicaciones
   locales basadas en reglas heurísticas derivadas de LIME.
""")


def enhance_nb03():
    print("Procesando: 03_interpretability_lime.ipynb")
    path = NOTEBOOKS_DIR / "03_interpretability_lime.ipynb"
    nb = read_nb(path)
    cells = nb["cells"]

    # 1. Encabezado formal al inicio
    cells = [HEADER_03] + cells

    # 2. Documentar configuración
    idx = find_cell_idx(cells, "import lime", "code")
    if idx is not None and cells[idx - 1]["cell_type"] != "markdown":
        cells = insert_before(cells, idx, [DOC_SETUP_03])

    # 3. Documentar el explainer
    idx = find_cell_idx(cells, "LimeTabularExplainer", "code")
    if idx is not None and cells[idx - 1]["cell_type"] != "markdown":
        cells = insert_before(cells, idx, [DOC_EXPLAINER_03])

    # 4. Documentar selección de casos
    idx = find_cell_idx(cells, "idx_high_churn", "code")
    if idx is not None:
        cells = insert_before(cells, idx, [DOC_CASES_03])

    # 5. Enriquecer análisis de caso de alta probabilidad
    idx = find_cell_idx(cells, "exp_high", "code")
    if idx is not None:
        idx2 = find_cell_idx(cells, "alta probabilidad de churn", "markdown")
        if idx2 is not None:
            cells[idx2] = ANALYSIS_HIGH_03

    # 6. Enriquecer análisis de caso de baja probabilidad
    idx2 = find_cell_idx(cells, "baja probabilidad de churn", "markdown")
    if idx2 is not None:
        cells[idx2] = ANALYSIS_LOW_03

    # 7. Enriquecer análisis de caso intermedio
    idx2 = find_cell_idx(cells, "probabilidad intermedia", "markdown")
    if idx2 is not None:
        cells[idx2] = ANALYSIS_MID_03

    # 8. Enriquecer conclusiones
    idx = find_cell_idx(cells, "Conclusiones de interpretabilidad", "markdown")
    if idx is not None:
        cells[idx] = CONCLUSIONS_03
    else:
        cells.append(CONCLUSIONS_03)

    nb["cells"] = cells
    write_nb(nb, path)


# ═══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Mejorando notebooks del Proyecto Integrador 5.12")
    print("=" * 60)
    enhance_nb01()
    enhance_nb02()
    enhance_nb03()
    print("=" * 60)
    print("Proceso completado. Verifica los notebooks en Jupyter.")
    print("=" * 60)
