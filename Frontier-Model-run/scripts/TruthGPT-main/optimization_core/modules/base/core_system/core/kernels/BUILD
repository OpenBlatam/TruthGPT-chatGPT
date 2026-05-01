load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "kernels",
    srcs = [
        "gpu_accelerator.py",
    ],
    deps = [
        "//core/common_runtime:common_runtime",
        "//core/platform:platform",
        "//core/util:util",
    ],
    visibility = ["//visibility:public"],
)

