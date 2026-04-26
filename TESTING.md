# IoT Spam Control - Testing Guide

## Unit Tests
```bash
python -m pytest tests/
```

## Integration Tests
- Test network packet capture
- Test spam detection algorithms
- Test ML model predictions

## Manual Testing
1. Start the application
2. Simulate network traffic
3. Verify spam detection accuracy
4. Check visualization dashboard

## Performance Tests
- Test with 1000 packets/second
- Verify detection latency under 50ms
- Check memory usage

## Code Coverage
- Target: 80% code coverage
- Run: coverage run -m pytest
- Generate report: coverage html
