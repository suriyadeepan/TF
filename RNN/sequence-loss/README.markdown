
**How to queue runners in your session, for batching?**

https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/#creating-threads-to-prefetch-using-queuerunner-objects
https://www.tensorflow.org/versions/r0.10/how_tos/threading_and_queues/#threading-and-queues

## Notes

- The sequence_length option only saves computational time for rnn, not dynamic_rnn
- [Batch Sequences](https://github.com/tensorflow/tensorflow/blob/4d579354034497d7fb58d01263dea72fc0014edd/tensorflow/g3doc/api_docs/python/functions_and_classes/shard4/tf.contrib.training.batch_sequences_with_states.md)
- [**How does batch\_matmul work?**](http://stackoverflow.com/questions/34183343/how-does-tensorflow-batch-matmul-work)
- This seems pretty handy 

```python
masktf.reshpae(masked_losses,tf.shape(y))
# tf.shape(y)
```


