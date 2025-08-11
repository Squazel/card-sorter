#!/usr/bin/env python
# Run all tests from the command line

import sys
import pytest

def main():
    """Run all tests for the card-sorter project."""
    # Run pytest with some common options
    args = [
        '--verbose',              # Detailed test output
        '--cov=card_ordering_rules.py',  # Coverage report for card_ordering_rules
        '--cov-report=term',      # Display coverage report in terminal
        '--no-header',            # Don't show pytest header
    ]
    
    # Allow passing additional arguments from command line
    args.extend(sys.argv[1:])
    
    # Run the tests
    return pytest.main(args)

if __name__ == '__main__':
    sys.exit(main())
