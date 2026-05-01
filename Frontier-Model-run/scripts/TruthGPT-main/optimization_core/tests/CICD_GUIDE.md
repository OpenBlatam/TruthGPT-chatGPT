# CI/CD Integration Guide

## Overview

This guide explains how to integrate the TruthGPT Optimization Core test framework with CI/CD pipelines.

## Features

✅ **Multi-platform testing**: Linux, Windows, macOS  
✅ **Multiple Python versions**: 3.8, 3.9, 3.10, 3.11  
✅ **Parallel execution**: Fast test runs  
✅ **Automated reporting**: HTML reports and JSON results  
✅ **Artifact storage**: Test results preserved  
✅ **PR comments**: Automatic feedback on pull requests  
✅ **Coverage tracking**: Codecov integration  

## GitHub Actions

### Pre-configured Workflow

The included `.github/workflows/tests.yml` provides:

1. **Automatic testing** on push/PR
2. **Matrix strategy** for multiple OS and Python versions
3. **Dependency caching** for faster builds
4. **Artifact upload** for test results
5. **Codecov integration** for coverage tracking
6. **PR comments** with test results

### Usage

The workflow is automatically triggered on:
- Push to `main` or `develop` branches
- Pull requests targeting these branches
- Weekly schedule (Sunday at midnight)

### Local Testing

```bash
# Install act (GitHub Actions locally)
# macOS
brew install act

# Linux/Windows (manual)
# Download from: https://github.com/nektos/act

# Run workflow locally
act -j test
```

## CI/CD Integration Examples

### GitHub Actions (Included)

```yaml
# .github/workflows/tests.yml (already created)
name: TruthGPT Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Run tests
        run: python tests/run_all_tests.py
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: test_results.json
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test

test:
  stage: test
  image: python:3.10
  script:
    - pip install -r requirements.txt
    - python tests/run_all_tests.py --save-results
  artifacts:
    paths:
      - test_results.json
      - test_report.html
    expire_in: 1 week
```

### Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    stages {
        stage('Test') {
            steps {
                sh '''
                    pip install -r requirements.txt
                    python tests/run_all_tests.py --save-results
                '''
            }
        }
        
        stage('Publish Results') {
            steps {
                publishHTML([
                    reportName: 'Test Report',
                    reportDir: '.',
                    reportFiles: 'test_report.html',
                    keepAll: true
                ])
            }
        }
    }
}
```

### CircleCI

```yaml
# .circleci/config.yml
version: 2.1

jobs:
  test:
    docker:
      - image: circleci/python:3.10
    steps:
      - checkout
      - run:
          name: Install dependencies
          command: pip install -r requirements.txt
      - run:
          name: Run tests
          command: python tests/run_all_tests.py --save-results
      - store_artifacts:
          path: test_results.json
          destination: test-results

workflows:
  main:
    jobs:
      - test
```

### Azure DevOps

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
      - main
      - develop

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.10'
  
  - script: |
      pip install -r requirements.txt
      python tests/run_all_tests.py --save-results
    displayName: 'Run Tests'
  
  - task: PublishTestResults@2
    inputs:
      testResultsFiles: 'test_results.json'
      testRunTitle: 'TruthGPT Tests'
  
  - task: PublishHTMLReports@1
    inputs:
      reportDir: '.'
      reportName: 'Test Report'
```

### Bitbucket Pipelines

```yaml
# bitbucket-pipelines.yml
image: python:3.10

pipelines:
  default:
    - step:
        name: Run Tests
        caches:
          - pip
        script:
          - pip install -r requirements.txt
          - python tests/run_all_tests.py --save-results
        artifacts:
          - test_results.json
          - test_report.html
```

## Customization

### Environment Variables

```bash
# Set custom test options via environment variables
export TRUTHGPT_TEST_PARALLEL=true
export TRUTHGPT_TEST_WORKERS=4
export TRUTHGPT_TEST_TIMEOUT=300
export TRUTHGPT_TEST_PATTERN=unit
```

### Custom Test Matrix

Modify `.github/workflows/tests.yml`:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest]
    python-version: ['3.9', '3.10']
    include:
      - os: ubuntu-latest
        python-version: '3.11'
        coverage: true
```

### Performance Thresholds

Add to your CI config:

```yaml
- name: Check performance
  run: |
    python -c "
    import json
    with open('test_results.json') as f:
        data = json.load(f)
    
    avg_time = data['performance_metrics']['average_execution_time']
    if avg_time > 5.0:
        print('⚠️ Performance regression!')
        exit(1)
    "
```

## Reporting

### HTML Reports

```bash
# Generate HTML report
python -c "
from tests.report_generator import HTMLReportGenerator
import json

with open('test_results.json') as f:
    results = json.load(f)

generator = HTMLReportGenerator()
generator.generate_report(results, 'test_report.html')
"
```

### Trend Analysis

```bash
# Analyze trends
python -c "
from tests.report_generator import TrendAnalyzer
import json

analyzer = TrendAnalyzer()

with open('test_results.json') as f:
    results = json.load(f)

analyzer.save_result(results)
analyzer.print_trends()
"
```

## Best Practices

### 1. Test Early and Often

```yaml
# Run tests on every push
on:
  push:
    branches: ['*']
```

### 2. Parallel Execution

```yaml
# Use pytest-xdist for parallel execution
- name: Install xdist
  run: pip install pytest-xdist

- name: Run tests in parallel
  run: pytest -n auto
```

### 3. Caching

```yaml
# Cache dependencies
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

### 4. Notifications

```yaml
# Notify on failure
- name: Notify on failure
  if: failure()
  run: |
    curl -X POST $SLACK_WEBHOOK \
      -d "text=Tests failed on ${{ github.ref }}"
```

### 5. Artifacts

```yaml
# Always upload artifacts
- uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: |
      test_results.json
      test_report.html
    retention-days: 30
```

## Troubleshooting

### Common Issues

#### Tests timeout in CI

```yaml
# Add timeout
- name: Run tests with timeout
  run: |
    timeout 600 python tests/run_all_tests.py
```

#### Memory issues

```yaml
# Limit parallel workers
- name: Run tests with limited workers
  run: |
    python tests/run_all_tests.py --max-workers=2
```

#### Import errors

```yaml
# Add to path
- name: Run tests
  run: |
    export PYTHONPATH=$PWD:$PYTHONPATH
    python tests/run_all_tests.py
```

## Monitoring

### Dashboard Example

Create a dashboard to track:
- Test success rates over time
- Performance trends
- Coverage trends
- Failed test patterns

### Slack Integration

```python
# slack_notifier.py
import json
import requests

def notify_slack(results, webhook_url):
    success = results['total_failures'] == 0
    
    payload = {
        "text": f"🧪 Tests {'✅ PASSED' if success else '❌ FAILED'}",
        "attachments": [{
            "color": "good" if success else "danger",
            "fields": [
                {"title": "Total Tests", "value": results['total_tests'], "short": True},
                {"title": "Success Rate", "value": f"{results['success_rate']:.1f}%", "short": True}
            ]
        }]
    }
    
    requests.post(webhook_url, json=payload)
```

## Summary

✅ **Easy integration** with major CI/CD platforms  
✅ **Automatic testing** on every push/PR  
✅ **Multiple platforms** and Python versions  
✅ **Rich reporting** with HTML and JSON  
✅ **Trend tracking** over time  
✅ **Best practices** documented  

The test framework is production-ready for CI/CD integration!


