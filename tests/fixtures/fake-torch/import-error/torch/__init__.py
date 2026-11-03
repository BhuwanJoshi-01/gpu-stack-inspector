"""Fake torch: a damaged install whose CUDA libraries are missing."""

raise ImportError("libcudnn.so.9: cannot open shared object file: No such file or directory")
