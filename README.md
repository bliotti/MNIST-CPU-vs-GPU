# Comparison of training time between MacOS and Windows WSL

The training time is significantly different between MacOS and Windows WSL. The training time on MacOS (CPU) is 5 minutes and 23 seconds, while on Windows WSL (GPU) is 27 seconds.

## CPU version of PyTorch

To run the CPU version of PyTorch, use the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## CUDA version of PyTorch

To run the CUDA version of PyTorch, use the following commands:

```bash
python3 -m venv venv-cuda
source venv-cuda/bin/activate
pip install -r requirements-cuda.txt
python main-cuda.py
```

### Training Time Comparison

#### MacOS

- **Version**: 15.2 (24C101)
- **Processor**: 2.3 GHz 8-Core Intel Core i9
- **Training Time**: 5 minutes and 23 seconds

```bash
(venv) [14:00:43] [indy] [~/ai/chatgpt-torch-example] ❱❱❱ time python3 main.py
Epoch [1/5], Loss: 0.1501
Epoch [2/5], Loss: 0.0401
Epoch [3/5], Loss: 0.0256
Epoch [4/5], Loss: 0.0170
Epoch [5/5], Loss: 0.0120
Accuracy on the test set: 98.83%
python3 main.py 2447.02s user 97.84s system 786% cpu 5:23.60 total
```

#### Windows WSL

- **OS**: Debian GNU/Linux 12 (bookworm)
- **GPU**: NVIDIA GeForce RTX 4060 Ti
- **Training Time**: 27 seconds

```bash
(env) [17:51:23] [bdlio@LiottiMachine] [~/kode/ai/training] ❱❱❱ time python main.py
Epoch [1/5], Loss: 0.1268
Epoch [2/5], Loss: 0.0399
Epoch [3/5], Loss: 0.0251
Epoch [4/5], Loss: 0.0172
Epoch [5/5], Loss: 0.0125
Accuracy on the test set: 99.06%
python main.py 30.27s user 1.21s system 115% cpu 27.179 total
```

### Visualizing Data

Optionally visualize or explore sample data with `MINST_RawFile.ipynb`.

```bash
(env) [17:51:23] [bdlio@LiottiMachine] [~/kode/ai/training] ❱❱❱ jupyter notebook MINST_RawFile.ipynb
```

## Conclusion

The training time on Windows WSL with a GPU is significantly faster compared to MacOS with a CPU. This demonstrates the advantage of using a GPU for deep learning tasks, as it can greatly reduce the time required for training neural networks.

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [WSL Documentation](https://docs.microsoft.com/en-us/windows/wsl/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
