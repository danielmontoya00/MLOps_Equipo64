#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Obesity Estimation
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 Module
	isort --check --diff Module
	black --check Module

## Format source code with black
.PHONY: format
format:
	isort Module
	black Module



## Run tests
.PHONY: test
test:
	python -m pytest tests

## Test monitoring system setup
.PHONY: test-monitoring
test-monitoring:
	@echo "Testing Data Drift Monitoring System..."
	@bash test_monitoring.sh

## Run data drift monitoring (default: mean_shift)
.PHONY: monitor-drift
monitor-drift:
	@echo "Running drift monitoring with scenario: $(SCENARIO)"
	$(PYTHON_INTERPRETER) -m models.monitor_drift --scenario $(or $(SCENARIO),mean_shift) --intensity $(or $(INTENSITY),0.3)

## Run all drift scenarios
.PHONY: monitor-all-scenarios
monitor-all-scenarios:
	@echo "Running all drift scenarios..."
	$(PYTHON_INTERPRETER) -m models.monitor_drift --scenario mean_shift --intensity 0.3
	$(PYTHON_INTERPRETER) -m models.monitor_drift --scenario variance_shift --intensity 0.4
	$(PYTHON_INTERPRETER) -m models.monitor_drift --scenario seasonal --intensity 0.5
	$(PYTHON_INTERPRETER) -m models.monitor_drift --scenario combined --intensity 0.3

## Generate drift simulation only
.PHONY: simulate-drift
simulate-drift:
	$(PYTHON_INTERPRETER) -c "from src.monitoring.drift_simulator import main; main()"

## Detect drift only (requires existing drifted data)
.PHONY: detect-drift
detect-drift:
	$(PYTHON_INTERPRETER) -c "from src.monitoring.drift_detector import main; main()"

## Monitor performance only
.PHONY: monitor-performance
monitor-performance:
	$(PYTHON_INTERPRETER) -c "from src.monitoring.performance_monitor import main; main()"
## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	aws s3 sync s3://bucket-name/data/ \
		data/ 
	

## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	aws s3 sync data/ \
		s3://bucket-name/data 
	



## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
