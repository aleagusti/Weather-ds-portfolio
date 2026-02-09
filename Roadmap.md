# Features
## F-00: Estado actual del proyecto (baseline)

Status: DONE

Descripción:
Conjunto de componentes ya implementados que definen la base funcional y estructural del proyecto. Esta feature actúa como baseline para el backlog a partir de este punto.

Alcance (todo DONE)
-   Arquitectura general del proyecto
-   Archivos base del repositorio
    -   'README.txt'
    -   '.gitignore'
    -   estructura de carpetas
-   Pipeline de ejecución del proyecto ('run_pipeline.sh')
-   Tests del proyecto '/tests' 
    - Test para asegurar que el proyecto corre
    - Validaciones básicas de resultados
-   Notebook de EDA inicial
-   'config.py' (orquestador de paths)
-   Notebook de EDA precipitacion
-   Notebook de Feature Engineering
-   'features.py' (orquestador de features)
-   'build_dataset.py' (orquestador de dataframe final)
-   Notebook de modelos base 
    - Placeholders: '05_xgboost'; '06_neural_networks'
-   Notebook de comparacion de modelos con metricas 
    - Placeholder: '07_model_comparison'
-   Script para correr modelo seleccionado 
    - Placeholder: 'models/selected_model.py' 

## F-01: Reproducibilidad y evaluación formal del modelado

Status: TO-DO

Objetivo

Asegurar que los resultados del modelado sean reproducibles, comparables y defendibles, separando claramente:

- entrenamiento

- evaluación

- selección de modelo

Esta feature no agrega “modelos nuevos”, sino criterio y disciplina sobre los existentes.

### US-01: Split temporal explícito y reproducible

Como reviewer técnico:
- Quiero que el split train/validation/test sea explícito y configurable
- Para evitar leakage y poder repetir resultados

Acceptance Criteria:
- El split es temporal, no aleatorio
- Las fechas/índices de corte están definidas en config
- El mismo split produce los mismos datasets en corridas sucesivas

### US-02: Métricas normalizadas y comparables

Como analista:
- Quiero calcular métricas homogéneas para todos los modelos
- Para poder compararlos sin ambigüedad

Acceptance Criteria:
- Se calculan al menos MAE, RMSE y R²
- Las métricas se calculan solo en validation/test
- Todas las métricas se guardan en una estructura común (df o csv)

### US-03: Notebook de comparación con criterio claro

Como reviewer:
- Quiero un notebook de comparación que muestre resultados consolidados
- Para entender por qué se elige un modelo y no otro

Acceptance Criteria:
- El notebook 07_model_comparison no entrena modelos
- Solo consume resultados ya generados
- Expone tablas + gráficos simples (no storytelling)

### US-04: Selección explícita de modelo ganador

Como usuario del pipeline:
- Quiero definir qué modelo es el “seleccionado”
- Para correr siempre el mismo en producción/experimentos

Acceptance Criteria:
- Existe una única fuente de verdad del modelo seleccionado
- models/selected_model.py usa esa definición
- Cambiar el modelo no requiere tocar lógica interna

Fuera de scope (explícito):
- No se agregan nuevos algoritmos
- No se optimizan hiperparámetros
- No se refactoriza EDA ni FE
- No se cambia F-00 salvo bug crítico

# Issues


