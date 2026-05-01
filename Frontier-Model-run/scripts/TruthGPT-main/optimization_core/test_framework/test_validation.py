"""
Test Validation Framework
Advanced validation testing for optimization core
"""

import unittest
import time
import logging
import random
import numpy as np
import re
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import sys
import os
from pathlib import Path

# Add the optimization core to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from test_framework.base_test import BaseTest, TestCategory, TestPriority

class ValidationTestType(Enum):
    """Validation test types."""
    INPUT_VALIDATION = "input_validation"
    OUTPUT_VALIDATION = "output_validation"
    DATA_VALIDATION = "data_validation"
    CONFIG_VALIDATION = "config_validation"
    SCHEMA_VALIDATION = "schema_validation"
    BUSINESS_VALIDATION = "business_validation"
    SECURITY_VALIDATION = "security_validation"
    PERFORMANCE_VALIDATION = "performance_validation"
    COMPATIBILITY_VALIDATION = "compatibility_validation"
    INTEGRITY_VALIDATION = "integrity_validation"

@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    description: str
    rule_type: str
    pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    required: bool = True
    custom_validator: Optional[callable] = None

@dataclass
class ValidationResult:
    """Validation result."""
    rule_name: str
    passed: bool
    message: str
    severity: str = "ERROR"
    suggestions: List[str] = field(default_factory=list)

@dataclass
class ValidationTestResult:
    """Validation test result."""
    test_type: ValidationTestType
    total_rules: int
    passed_rules: int
    failed_rules: int
    warning_rules: int
    validation_score: float
    results: List[ValidationResult] = field(default_factory=list)

class TestInputValidation(BaseTest):
    """Test input validation scenarios."""
    
    def setUp(self):
        super().setUp()
        self.input_rules = [
            ValidationRule("email_validation", "Email format validation", "regex", r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
            ValidationRule("numeric_range", "Numeric range validation", "range", min_value=0, max_value=100),
            ValidationRule("string_length", "String length validation", "length", min_value=1, max_value=255),
            ValidationRule("required_field", "Required field validation", "required"),
            ValidationRule("alphanumeric", "Alphanumeric validation", "regex", r"^[a-zA-Z0-9]+$")
        ]
        self.input_results = []
    
    def test_email_validation(self):
        """Test email validation."""
        rule = self.input_rules[0]
        test_emails = [
            "valid@example.com",
            "test.user@domain.co.uk",
            "invalid-email",
            "@invalid.com",
            "user@",
            "user@domain",
            "user@domain.",
            "user@domain.com."
        ]
        
        results = []
        for email in test_emails:
            is_valid = bool(re.match(rule.pattern, email))
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=is_valid,
                message=f"Email '{email}' {'valid' if is_valid else 'invalid'}",
                severity="ERROR" if not is_valid else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.INPUT_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.input_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 2 valid emails out of 8
        self.assertEqual(passed_count, 2)
    
    def test_numeric_range_validation(self):
        """Test numeric range validation."""
        rule = self.input_rules[1]
        test_values = [0, 50, 100, -1, 101, 25.5, 75.8, 150]
        
        results = []
        for value in test_values:
            is_valid = rule.min_value <= value <= rule.max_value
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=is_valid,
                message=f"Value {value} {'within' if is_valid else 'outside'} range [{rule.min_value}, {rule.max_value}]",
                severity="ERROR" if not is_valid else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.INPUT_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.input_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 4 valid values out of 8
        self.assertEqual(passed_count, 4)
    
    def test_string_length_validation(self):
        """Test string length validation."""
        rule = self.input_rules[2]
        test_strings = [
            "a",  # min length
            "valid string",
            "a" * 255,  # max length
            "",  # too short
            "a" * 256,  # too long
            "normal string",
            "x" * 100,
            "y" * 200
        ]
        
        results = []
        for string in test_strings:
            is_valid = rule.min_value <= len(string) <= rule.max_value
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=is_valid,
                message=f"String length {len(string)} {'within' if is_valid else 'outside'} range [{rule.min_value}, {rule.max_value}]",
                severity="ERROR" if not is_valid else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.INPUT_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.input_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 5 valid strings out of 8
        self.assertEqual(passed_count, 5)
    
    def test_required_field_validation(self):
        """Test required field validation."""
        rule = self.input_rules[3]
        test_values = [
            "valid value",
            "",
            None,
            "   ",  # whitespace only
            "another valid value",
            "\t\n",  # whitespace characters
            "0",
            False
        ]
        
        results = []
        for value in test_values:
            is_valid = value is not None and str(value).strip() != ""
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=is_valid,
                message=f"Value '{value}' {'present' if is_valid else 'missing or empty'}",
                severity="ERROR" if not is_valid else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.INPUT_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.input_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 4 valid values out of 8
        self.assertEqual(passed_count, 4)
    
    def test_alphanumeric_validation(self):
        """Test alphanumeric validation."""
        rule = self.input_rules[4]
        test_strings = [
            "valid123",
            "ABC123",
            "test",
            "123",
            "invalid-string",
            "invalid_string",
            "invalid.string",
            "valid123ABC"
        ]
        
        results = []
        for string in test_strings:
            is_valid = bool(re.match(rule.pattern, string))
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=is_valid,
                message=f"String '{string}' {'alphanumeric' if is_valid else 'contains non-alphanumeric characters'}",
                severity="ERROR" if not is_valid else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.INPUT_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.input_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 5 valid strings out of 8
        self.assertEqual(passed_count, 5)
    
    def get_input_validation_metrics(self) -> Dict[str, Any]:
        """Get input validation metrics."""
        total_rules = len(self.input_results)
        passed_rules = len([r for r in self.input_results if r['status'] == 'PASS'])
        
        if total_rules == 0:
            return {}
        
        avg_validation_score = sum(r['result'].validation_score for r in self.input_results) / total_rules
        total_passed_rules = sum(r['result'].passed_rules for r in self.input_results)
        total_failed_rules = sum(r['result'].failed_rules for r in self.input_results)
        
        return {
            'total_rules': total_rules,
            'passed_rules': passed_rules,
            'success_rate': (passed_rules / total_rules * 100),
            'average_validation_score': avg_validation_score,
            'total_passed_validations': total_passed_rules,
            'total_failed_validations': total_failed_rules,
            'input_validation_quality': 'EXCELLENT' if avg_validation_score > 0.9 else 'GOOD' if avg_validation_score > 0.8 else 'FAIR' if avg_validation_score > 0.7 else 'POOR'
        }

class TestOutputValidation(BaseTest):
    """Test output validation scenarios."""
    
    def setUp(self):
        super().setUp()
        self.output_rules = [
            ValidationRule("json_format", "JSON format validation", "format"),
            ValidationRule("xml_format", "XML format validation", "format"),
            ValidationRule("numeric_output", "Numeric output validation", "type"),
            ValidationRule("string_output", "String output validation", "type"),
            ValidationRule("array_output", "Array output validation", "type")
        ]
        self.output_results = []
    
    def test_json_format_validation(self):
        """Test JSON format validation."""
        rule = self.output_rules[0]
        test_outputs = [
            '{"key": "value"}',
            '{"array": [1, 2, 3]}',
            '{"nested": {"key": "value"}}',
            'invalid json',
            '{"incomplete": }',
            '{"valid": "json", "number": 123}',
            'not json at all',
            '{"empty": ""}'
        ]
        
        results = []
        for output in test_outputs:
            try:
                json.loads(output)
                is_valid = True
            except (json.JSONDecodeError, TypeError):
                is_valid = False
            
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=is_valid,
                message=f"JSON output {'valid' if is_valid else 'invalid'}",
                severity="ERROR" if not is_valid else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.OUTPUT_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.output_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 5 valid JSON outputs out of 8
        self.assertEqual(passed_count, 5)
    
    def test_xml_format_validation(self):
        """Test XML format validation."""
        rule = self.output_rules[1]
        test_outputs = [
            '<root><item>value</item></root>',
            '<root><item>value</item><item>another</item></root>',
            '<root><nested><item>value</item></nested></root>',
            'invalid xml',
            '<root><item>value</item>',  # unclosed tag
            '<root><item>value</item></root>',
            'not xml at all',
            '<root></root>'
        ]
        
        results = []
        for output in test_outputs:
            try:
                ET.fromstring(output)
                is_valid = True
            except ET.ParseError:
                is_valid = False
            
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=is_valid,
                message=f"XML output {'valid' if is_valid else 'invalid'}",
                severity="ERROR" if not is_valid else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.OUTPUT_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.output_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 4 valid XML outputs out of 8
        self.assertEqual(passed_count, 4)
    
    def test_numeric_output_validation(self):
        """Test numeric output validation."""
        rule = self.output_rules[2]
        test_outputs = [
            123,
            45.67,
            "123",
            "45.67",
            "not a number",
            None,
            [1, 2, 3],
            {"key": "value"}
        ]
        
        results = []
        for output in test_outputs:
            is_valid = isinstance(output, (int, float)) or (isinstance(output, str) and output.replace('.', '').replace('-', '').isdigit())
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=is_valid,
                message=f"Output {output} {'numeric' if is_valid else 'not numeric'}",
                severity="ERROR" if not is_valid else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.OUTPUT_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.output_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 4 valid numeric outputs out of 8
        self.assertEqual(passed_count, 4)
    
    def test_string_output_validation(self):
        """Test string output validation."""
        rule = self.output_rules[3]
        test_outputs = [
            "valid string",
            "another string",
            123,
            45.67,
            None,
            ["list", "of", "strings"],
            {"key": "value"},
            ""
        ]
        
        results = []
        for output in test_outputs:
            is_valid = isinstance(output, str)
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=is_valid,
                message=f"Output {output} {'string' if is_valid else 'not string'}",
                severity="ERROR" if not is_valid else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.OUTPUT_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.output_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 3 valid string outputs out of 8
        self.assertEqual(passed_count, 3)
    
    def test_array_output_validation(self):
        """Test array output validation."""
        rule = self.output_rules[4]
        test_outputs = [
            [1, 2, 3],
            ["a", "b", "c"],
            [],
            [{"key": "value"}],
            "not an array",
            123,
            None,
            {"key": "value"}
        ]
        
        results = []
        for output in test_outputs:
            is_valid = isinstance(output, list)
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=is_valid,
                message=f"Output {output} {'array' if is_valid else 'not array'}",
                severity="ERROR" if not is_valid else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.OUTPUT_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.output_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 4 valid array outputs out of 8
        self.assertEqual(passed_count, 4)
    
    def get_output_validation_metrics(self) -> Dict[str, Any]:
        """Get output validation metrics."""
        total_rules = len(self.output_results)
        passed_rules = len([r for r in self.output_results if r['status'] == 'PASS'])
        
        if total_rules == 0:
            return {}
        
        avg_validation_score = sum(r['result'].validation_score for r in self.output_results) / total_rules
        total_passed_rules = sum(r['result'].passed_rules for r in self.output_results)
        total_failed_rules = sum(r['result'].failed_rules for r in self.output_results)
        
        return {
            'total_rules': total_rules,
            'passed_rules': passed_rules,
            'success_rate': (passed_rules / total_rules * 100),
            'average_validation_score': avg_validation_score,
            'total_passed_validations': total_passed_rules,
            'total_failed_validations': total_failed_rules,
            'output_validation_quality': 'EXCELLENT' if avg_validation_score > 0.9 else 'GOOD' if avg_validation_score > 0.8 else 'FAIR' if avg_validation_score > 0.7 else 'POOR'
        }

class TestDataValidation(BaseTest):
    """Test data validation scenarios."""
    
    def setUp(self):
        super().setUp()
        self.data_rules = [
            ValidationRule("data_integrity", "Data integrity validation", "integrity"),
            ValidationRule("data_consistency", "Data consistency validation", "consistency"),
            ValidationRule("data_completeness", "Data completeness validation", "completeness"),
            ValidationRule("data_accuracy", "Data accuracy validation", "accuracy"),
            ValidationRule("data_timeliness", "Data timeliness validation", "timeliness")
        ]
        self.data_results = []
    
    def test_data_integrity_validation(self):
        """Test data integrity validation."""
        rule = self.data_rules[0]
        
        # Simulate data integrity checks
        integrity_checks = [
            {"check": "primary_key_uniqueness", "passed": True, "count": 1000},
            {"check": "foreign_key_constraints", "passed": True, "count": 500},
            {"check": "data_type_consistency", "passed": False, "count": 50},
            {"check": "null_constraint_violations", "passed": True, "count": 0},
            {"check": "duplicate_records", "passed": False, "count": 25}
        ]
        
        results = []
        for check in integrity_checks:
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=check["passed"],
                message=f"Integrity check '{check['check']}' {'passed' if check['passed'] else 'failed'} ({check['count']} issues)",
                severity="ERROR" if not check["passed"] else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.DATA_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.data_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 3 passed checks out of 5
        self.assertEqual(passed_count, 3)
    
    def test_data_consistency_validation(self):
        """Test data consistency validation."""
        rule = self.data_rules[1]
        
        # Simulate data consistency checks
        consistency_checks = [
            {"check": "cross_table_consistency", "passed": True, "inconsistencies": 0},
            {"check": "temporal_consistency", "passed": False, "inconsistencies": 15},
            {"check": "logical_consistency", "passed": True, "inconsistencies": 0},
            {"check": "referential_consistency", "passed": False, "inconsistencies": 8},
            {"check": "business_rule_consistency", "passed": True, "inconsistencies": 0}
        ]
        
        results = []
        for check in consistency_checks:
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=check["passed"],
                message=f"Consistency check '{check['check']}' {'passed' if check['passed'] else 'failed'} ({check['inconsistencies']} inconsistencies)",
                severity="ERROR" if not check["passed"] else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.DATA_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.data_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 3 passed checks out of 5
        self.assertEqual(passed_count, 3)
    
    def test_data_completeness_validation(self):
        """Test data completeness validation."""
        rule = self.data_rules[2]
        
        # Simulate data completeness checks
        completeness_checks = [
            {"field": "user_id", "completeness": 100.0, "passed": True},
            {"field": "email", "completeness": 95.5, "passed": True},
            {"field": "phone", "completeness": 78.2, "passed": False},
            {"field": "address", "completeness": 88.7, "passed": True},
            {"field": "preferences", "completeness": 45.3, "passed": False}
        ]
        
        results = []
        for check in completeness_checks:
            threshold = 80.0
            passed = check["completeness"] >= threshold
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=passed,
                message=f"Field '{check['field']}' completeness {check['completeness']:.1f}% {'meets' if passed else 'below'} threshold ({threshold}%)",
                severity="WARNING" if not passed else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.DATA_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.data_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 3 passed checks out of 5
        self.assertEqual(passed_count, 3)
    
    def test_data_accuracy_validation(self):
        """Test data accuracy validation."""
        rule = self.data_rules[3]
        
        # Simulate data accuracy checks
        accuracy_checks = [
            {"check": "email_format_accuracy", "accuracy": 98.5, "passed": True},
            {"check": "phone_format_accuracy", "accuracy": 92.3, "passed": True},
            {"check": "date_format_accuracy", "accuracy": 85.7, "passed": True},
            {"check": "numeric_range_accuracy", "accuracy": 76.4, "passed": False},
            {"check": "categorical_accuracy", "accuracy": 88.9, "passed": True}
        ]
        
        results = []
        for check in accuracy_checks:
            threshold = 80.0
            passed = check["accuracy"] >= threshold
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=passed,
                message=f"Accuracy check '{check['check']}' {check['accuracy']:.1f}% {'meets' if passed else 'below'} threshold ({threshold}%)",
                severity="WARNING" if not passed else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.DATA_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.data_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 4 passed checks out of 5
        self.assertEqual(passed_count, 4)
    
    def test_data_timeliness_validation(self):
        """Test data timeliness validation."""
        rule = self.data_rules[4]
        
        # Simulate data timeliness checks
        timeliness_checks = [
            {"check": "real_time_data_freshness", "age_hours": 0.5, "passed": True},
            {"check": "hourly_data_freshness", "age_hours": 2.3, "passed": True},
            {"check": "daily_data_freshness", "age_hours": 18.7, "passed": True},
            {"check": "weekly_data_freshness", "age_hours": 168.5, "passed": False},
            {"check": "monthly_data_freshness", "age_hours": 720.0, "passed": False}
        ]
        
        results = []
        for check in timeliness_checks:
            # Define freshness thresholds
            thresholds = {
                "real_time_data_freshness": 1.0,
                "hourly_data_freshness": 4.0,
                "daily_data_freshness": 24.0,
                "weekly_data_freshness": 168.0,
                "monthly_data_freshness": 720.0
            }
            
            threshold = thresholds.get(check["check"], 24.0)
            passed = check["age_hours"] <= threshold
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=passed,
                message=f"Timeliness check '{check['check']}' age {check['age_hours']:.1f}h {'within' if passed else 'exceeds'} threshold ({threshold}h)",
                severity="WARNING" if not passed else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.DATA_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.data_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 3 passed checks out of 5
        self.assertEqual(passed_count, 3)
    
    def get_data_validation_metrics(self) -> Dict[str, Any]:
        """Get data validation metrics."""
        total_rules = len(self.data_results)
        passed_rules = len([r for r in self.data_results if r['status'] == 'PASS'])
        
        if total_rules == 0:
            return {}
        
        avg_validation_score = sum(r['result'].validation_score for r in self.data_results) / total_rules
        total_passed_rules = sum(r['result'].passed_rules for r in self.data_results)
        total_failed_rules = sum(r['result'].failed_rules for r in self.data_results)
        
        return {
            'total_rules': total_rules,
            'passed_rules': passed_rules,
            'success_rate': (passed_rules / total_rules * 100),
            'average_validation_score': avg_validation_score,
            'total_passed_validations': total_passed_rules,
            'total_failed_validations': total_failed_rules,
            'data_validation_quality': 'EXCELLENT' if avg_validation_score > 0.9 else 'GOOD' if avg_validation_score > 0.8 else 'FAIR' if avg_validation_score > 0.7 else 'POOR'
        }

class TestConfigValidation(BaseTest):
    """Test configuration validation scenarios."""
    
    def setUp(self):
        super().setUp()
        self.config_rules = [
            ValidationRule("config_schema", "Configuration schema validation", "schema"),
            ValidationRule("config_values", "Configuration values validation", "values"),
            ValidationRule("config_dependencies", "Configuration dependencies validation", "dependencies"),
            ValidationRule("config_security", "Configuration security validation", "security"),
            ValidationRule("config_performance", "Configuration performance validation", "performance")
        ]
        self.config_results = []
    
    def test_config_schema_validation(self):
        """Test configuration schema validation."""
        rule = self.config_rules[0]
        
        # Simulate configuration schema validation
        config_schemas = [
            {"name": "database_config", "valid": True, "errors": []},
            {"name": "api_config", "valid": False, "errors": ["missing required field: timeout"]},
            {"name": "logging_config", "valid": True, "errors": []},
            {"name": "cache_config", "valid": False, "errors": ["invalid field: unknown_field"]},
            {"name": "security_config", "valid": True, "errors": []}
        ]
        
        results = []
        for schema in config_schemas:
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=schema["valid"],
                message=f"Schema '{schema['name']}' {'valid' if schema['valid'] else 'invalid'}: {', '.join(schema['errors']) if schema['errors'] else 'no errors'}",
                severity="ERROR" if not schema["valid"] else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.CONFIG_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.config_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 3 valid schemas out of 5
        self.assertEqual(passed_count, 3)
    
    def test_config_values_validation(self):
        """Test configuration values validation."""
        rule = self.config_rules[1]
        
        # Simulate configuration values validation
        config_values = [
            {"name": "port", "value": 8080, "valid": True, "reason": "within valid range"},
            {"name": "timeout", "value": -1, "valid": False, "reason": "negative value not allowed"},
            {"name": "max_connections", "value": 1000, "valid": True, "reason": "within valid range"},
            {"name": "debug_mode", "value": "invalid", "valid": False, "reason": "must be boolean"},
            {"name": "log_level", "value": "INFO", "valid": True, "reason": "valid enum value"}
        ]
        
        results = []
        for config in config_values:
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=config["valid"],
                message=f"Config '{config['name']}' value {config['value']} {'valid' if config['valid'] else 'invalid'}: {config['reason']}",
                severity="ERROR" if not config["valid"] else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.CONFIG_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.config_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 3 valid values out of 5
        self.assertEqual(passed_count, 3)
    
    def test_config_dependencies_validation(self):
        """Test configuration dependencies validation."""
        rule = self.config_rules[2]
        
        # Simulate configuration dependencies validation
        dependency_checks = [
            {"dependency": "database_url -> database_config", "satisfied": True, "reason": "all required fields present"},
            {"dependency": "api_key -> authentication", "satisfied": False, "reason": "api_key missing"},
            {"dependency": "ssl_cert -> ssl_enabled", "satisfied": True, "reason": "ssl enabled and cert provided"},
            {"dependency": "cache_size -> cache_enabled", "satisfied": False, "reason": "cache enabled but size not set"},
            {"dependency": "log_file -> logging_enabled", "satisfied": True, "reason": "logging enabled and file specified"}
        ]
        
        results = []
        for check in dependency_checks:
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=check["satisfied"],
                message=f"Dependency '{check['dependency']}' {'satisfied' if check['satisfied'] else 'not satisfied'}: {check['reason']}",
                severity="ERROR" if not check["satisfied"] else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.CONFIG_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.config_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 3 satisfied dependencies out of 5
        self.assertEqual(passed_count, 3)
    
    def test_config_security_validation(self):
        """Test configuration security validation."""
        rule = self.config_rules[3]
        
        # Simulate configuration security validation
        security_checks = [
            {"check": "password_strength", "secure": True, "reason": "strong password policy enforced"},
            {"check": "ssl_configuration", "secure": True, "reason": "SSL properly configured"},
            {"check": "api_key_exposure", "secure": False, "reason": "API key in plain text"},
            {"check": "database_credentials", "secure": True, "reason": "credentials encrypted"},
            {"check": "debug_mode", "secure": False, "reason": "debug mode enabled in production"}
        ]
        
        results = []
        for check in security_checks:
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=check["secure"],
                message=f"Security check '{check['check']}' {'secure' if check['secure'] else 'insecure'}: {check['reason']}",
                severity="ERROR" if not check["secure"] else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.CONFIG_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.config_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 3 secure configurations out of 5
        self.assertEqual(passed_count, 3)
    
    def test_config_performance_validation(self):
        """Test configuration performance validation."""
        rule = self.config_rules[4]
        
        # Simulate configuration performance validation
        performance_checks = [
            {"check": "connection_pool_size", "optimal": True, "reason": "pool size within recommended range"},
            {"check": "cache_size", "optimal": True, "reason": "cache size appropriate for workload"},
            {"check": "timeout_values", "optimal": False, "reason": "timeout too high, may cause delays"},
            {"check": "thread_count", "optimal": True, "reason": "thread count matches CPU cores"},
            {"check": "memory_allocation", "optimal": False, "reason": "memory allocation exceeds available"}
        ]
        
        results = []
        for check in performance_checks:
            results.append(ValidationResult(
                rule_name=rule.name,
                passed=check["optimal"],
                message=f"Performance check '{check['check']}' {'optimal' if check['optimal'] else 'suboptimal'}: {check['reason']}",
                severity="WARNING" if not check["optimal"] else "INFO"
            ))
        
        passed_count = len([r for r in results if r.passed])
        total_count = len(results)
        validation_score = passed_count / total_count
        
        result = ValidationTestResult(
            test_type=ValidationTestType.CONFIG_VALIDATION,
            total_rules=total_count,
            passed_rules=passed_count,
            failed_rules=total_count - passed_count,
            warning_rules=0,
            validation_score=validation_score,
            results=results
        )
        
        self.config_results.append({
            'rule': rule.name,
            'result': result,
            'status': 'PASS'
        })
        
        # Expected: 3 optimal configurations out of 5
        self.assertEqual(passed_count, 3)
    
    def get_config_validation_metrics(self) -> Dict[str, Any]:
        """Get configuration validation metrics."""
        total_rules = len(self.config_results)
        passed_rules = len([r for r in self.config_results if r['status'] == 'PASS'])
        
        if total_rules == 0:
            return {}
        
        avg_validation_score = sum(r['result'].validation_score for r in self.config_results) / total_rules
        total_passed_rules = sum(r['result'].passed_rules for r in self.config_results)
        total_failed_rules = sum(r['result'].failed_rules for r in self.config_results)
        
        return {
            'total_rules': total_rules,
            'passed_rules': passed_rules,
            'success_rate': (passed_rules / total_rules * 100),
            'average_validation_score': avg_validation_score,
            'total_passed_validations': total_passed_rules,
            'total_failed_validations': total_failed_rules,
            'config_validation_quality': 'EXCELLENT' if avg_validation_score > 0.9 else 'GOOD' if avg_validation_score > 0.8 else 'FAIR' if avg_validation_score > 0.7 else 'POOR'
        }

if __name__ == '__main__':
    unittest.main()










