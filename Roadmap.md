# Features
## F-00: Current Project State (Baseline)

Status: *DONE*

Description:
Set of components already implemented that define the functional and structural foundation of the project. This feature acts as the baseline for the backlog from this point forward.

Scope (all DONE)
- General project architecture
- Base repository files
    - 'README.txt'
    - '.gitignore'
    - folder structure
- Project execution pipeline ('run_pipeline.sh')
- Project tests '/tests'
    - Test to ensure the project runs
    - Basic result validations
- Initial EDA notebook
- 'config.py' (paths orchestrator)
- Precipitation EDA notebook
- Feature Engineering notebook
- 'features.py' (feature orchestrator)
- 'build_dataset.py' (final dataframe orchestrator)
- Base models notebook
    - Placeholders: '05_xgboost'; '06_neural_networks'
- Model comparison notebook with metrics
    - Placeholder: '07_model_comparison'
- Script to run selected model
    - Placeholder: 'models/selected_model.py'

## F-01: Reproducibility and Formal Modeling Evaluation

Status: *DONE*

Objective:
Ensure modeling results are reproducible, comparable, and defensible by clearly stating:
- training
- evaluation
- model selection

This feature does not add “new models,” but rather introduces rigor and discipline to the existing ones.

### US-01: Dynamic and reproducible temporal split

Status: *DONE*

As a technical reviewer:
- I want the train/test split to be temporal and configurable
- To avoid leakage and ensure methodological consistency

Acceptance Criteria:
- The split is strictly temporal (no shuffle)
- The test set corresponds to the last X% of the dataset
- The test percentage is defined in config
- The dataset is explicitly ordered by date before the split
- Successive runs with the same dataset produce the same split

### US-02: Standardized and comparable metrics

Status: *DONE*

As an analyst:
- I want homogeneous metrics calculated for all models
- To compare them without ambiguity

Acceptance Criteria:
- At least MAE, RMSE, and R² are calculated
- Metrics are calculated only on validation/test
- All metrics are saved in a common structure (df or csv)

### US-03: Comparison notebook with clear criteria

Status: *DONE*

As a reviewer:
- I want a comparison notebook that shows consolidated results
- To understand why one model is chosen over another

Acceptance Criteria:
- Notebook 07_model_comparison does not train models
- It only consumes already generated results
- It exposes tables + simple charts (no storytelling)

### US-04: Explicit winning model selection

Status: *DONE*

As a pipeline user:
- I want to define which model is the “selected” one
- To always run the same model in production/experiments

Acceptance Criteria:
- There is a single source of truth for the selected model
- models/selected_model.py uses that definition
- Changing the model does not require modifying internal logic

Out of scope (explicit):
- No new algorithms are added
- No hyperparameters are optimized
- No EDA or Feature Engineering refactoring
- F-00 is not modified except for critical bugs

## F-02: Model Expansion: XGBoost & Neural Networks

Status: *DONE*

Objective: Extend the current pipeline by incorporating:
- XGBoost
- Neural Network (MLP)

Maintaining:
- Same temporal split
- Same evaluation function
- Same result structure
- Same output format

### US-01: Reproducible XGBoost notebook

Status: *DONE*

As a technical reviewer:
- I want a dedicated XGBoost notebook
- To evaluate its performance under the same experimental framework as the base model

Acceptance Criteria:
- Uses the same temporal split defined in config
- Uses src/evaluation.py for metrics
- Normalizes features if required by the model
- Does not alter metric scale
- Saves results in a format identical to ablation_results_vXXX.csv
- Does not duplicate evaluation logic
- Allows minimal hyperparameter grid (max_depth, n_estimators, learning_rate)
- Documents assumptions and decisions
- Includes explicit random_state control

### US-02: Reproducible Neural Network notebook

Status: *DONE*

As a technical reviewer:
- I want a dedicated Neural Network (MLP) notebook
- To evaluate nonlinear models under the same framework

Acceptance Criteria:
- Uses the same temporal split defined in config
- Uses src/evaluation.py for metrics
- Normalizes features if required by the model
- Does not alter metric scale
- Saves results in a format identical to ablation_results_vXXX.csv
- Does not duplicate evaluation logic
- Allows minimal hyperparameter grid
- Documents assumptions and decisions
- Includes explicit random_state control

### US-03: Structural compatibility

Status: *DONE*

As a technical reviewer:
- I want new notebooks to produce outputs compatible with Notebook 07
- To compare all models without additional transformation

Acceptance Criteria:
- Generated CSVs have exactly the same columns
- No model-specific extra metrics
- No missing columns
- pd.concat() works without errors

## F-03: Model Improvement Iterations

Status: *Planned*

Objective:
- Explore structural improvements to:
- Improve precipitation spike prediction
- Reduce bias on heavy rainfall days
- Incorporate explicit physical knowledge

### US-01: Log-transformed target evaluation

Status: *Not started*

As a model researcher:
- I want to evaluate the impact of using log1p(target)
- To analyze whether it improves extreme event prediction

Acceptance Criteria:
- The same ablation grid is executed with USE_LOG_TARGET=True
- Metrics are calculated on the original scale (mm)
- Results are saved with differentiated EXP_VERSION
- Compared against baseline in Notebook 07

### US-02: Two-stage rainfall modeling

Status: *Not started*

As a pipeline designer:
- I want to evaluate a two-stage approach (classification + regression)
- To improve modeling of rainy vs non-rainy days

Acceptance Criteria:
- A binary variable is_rain is defined
- A classification model is trained
- A regression model is trained only on rainy days
- Final output is combined
- Metrics are reported comparably with baseline

### US-03: Physically-motivated feature interactions

Status: *Not started*

As a physical modeler:
- I want to incorporate relevant meteorological interactions
- To evaluate whether physical knowledge improves performance

Acceptance Criteria:
- New features are implemented in src/features.py
- Their physical motivation is documented
- They are added to new experiments in EXPERIMENTS
- The ablation grid is executed again
- Results are compared against baseline

## F-04: Production Robustness & Data Resilience

Status: *Planned*

Objective:
Strengthen the pipeline against real-world production scenarios, including:
- API rate limits
- Schema drift
- Corrupted or unexpected data
- Intermittent failures
- Structural input validation

The goal is not to improve model metrics, but to improve system reliability and resilience.

### US-01: API Rate Limit Handling

Status: *Not started*

As a systems designer:
- I want the data fetching layer to properly handle API rate limits
- So that automated pipeline executions do not fail unexpectedly

Acceptance Criteria:
- Explicit handling of HTTP 429 responses
- Retry mechanism with exponential backoff
- Configurable timeout
- Logging of failed attempts
- Controlled failure after maximum retries

### US-02: Schema Drift Detection

Status: *Not started*

As a data engineer:
- I want to validate that the dataset schema matches expectations
- So that silent upstream changes do not corrupt the modeling pipeline

Acceptance Criteria:
- Explicit list of required columns
- Schema validation before feature engineering
- Clear error message if required columns are missing
- Optional validation of expected data types
- Unit tests simulating schema drift

### US-03: Data Integrity Validation

Status: *Not started*

As the pipeline owner:
- I want to validate that input data respects basic physical constraints
- So that corrupted data does not propagate into the model

Acceptance Criteria:
- No negative precipitation values
- No duplicated timestamps
- No infinite values
- Basic sanity checks on plausible ranges
- Unit tests validating integrity checks

### US-04: Structured Logging

Status: *Not started*

As a system operator:
- I want structured logging instead of ad-hoc print statements
- So that monitoring and debugging are easier in production environments

Acceptance Criteria:
- Use of Python’s logging module
- Consistent log levels (INFO, WARNING, ERROR)
- No raw print statements in production code
- Meaningful and structured log messages

### US-05: Failure Simulation Tests

Status: *Not started*

As a technical reviewer:
- I want tests that simulate real-world failures
- So that pipeline resilience can be validated automatically

Acceptance Criteria:
- Mocked API returning HTTP 429
- Mocked incomplete or malformed JSON
- Mocked schema change
- Tests fail if validation mechanisms do not trigger

# Issues


