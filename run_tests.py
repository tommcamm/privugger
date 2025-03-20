#!/usr/bin/env python
"""
Test runner script for privugger.

This script provides a simple way to run the test suite for privugger
across different platforms (Windows, Linux, macOS).

Usage:
    python run_tests.py [options]

Options:
    --all           Run all tests
    --pyro          Run only pyro backend tests
    --specific=PATH Run a specific test file or test class
    --fast          Skip tests marked as slow
    --help          Show this help message
"""

import os
import sys
import subprocess
import argparse


def run_tests(args):
    """Run tests based on command line arguments."""
    # Base pytest command
    cmd = ["pytest"]
    
    # Add verbosity
    cmd.append("-v")
    
    # Handle specific test selections
    if args.all:
        # Run all tests
        pass  # No additional arguments needed for all tests
    elif args.pyro:
        # Run only pyro tests
        cmd.append("-m")
        cmd.append("pyro")
    elif args.specific:
        # Run a specific test file or class
        cmd.append(args.specific)
    
    # Skip slow tests if requested
    if args.fast:
        cmd.append("-m")
        cmd.append("not slow")
    
    # Print the command being run
    print(f"Running: {' '.join(cmd)}")
    
    # Run the tests
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run privugger tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Define the mutually exclusive test selection options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--all", action="store_true", help="Run all tests")
    test_group.add_argument("--pyro", action="store_true", help="Run only pyro backend tests")
    test_group.add_argument("--specific", help="Run a specific test file or test class")
    
    # Other options
    parser.add_argument("--fast", action="store_true", help="Skip tests marked as slow")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Default to --all if no test selection option was provided
    if not (args.all or args.pyro or args.specific):
        args.all = True
    
    # Run the tests
    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()
