load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "ops",
    srcs = [
        "ultra_fast_optimizer.py",
        "extreme_optimizer.py",
        "quantum_extreme_optimizer.py",
    ],
    deps = [
        "//core/common_runtime:common_runtime",
        "//core/framework:framework",
        "//core/kernels:kernels",
        "//core/util:util",
    ],
    visibility = ["//visibility:public"],
)

