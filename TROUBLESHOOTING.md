# IoT Spam Control - Troubleshooting

## Common Issues

### Issue: Module not found
**Solution:** Run pip install -r requirements.txt

### Issue: Database connection failed
**Solution:** Check DATABASE_URL environment variable and network connectivity.

### Issue: ML model loading error
**Solution:** Verify the model file exists and is not corrupted. Re-train if needed.

### Issue: Visualization not updating
**Solution:** Refresh browser, check WebSocket connection, verify server is running.

### Issue: High memory usage
**Solution:** Reduce batch size in config, limit packet capture window.

### Issue: False positives in spam detection
**Solution:** Retrain ML model with updated dataset, adjust spam threshold.
