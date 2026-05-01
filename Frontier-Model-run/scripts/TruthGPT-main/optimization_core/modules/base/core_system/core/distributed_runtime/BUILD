load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "distributed_runtime",
    srcs = [
        "distributed_optimizer.py",
    ],
    deps = [
        "//core/common_runtime:common_runtime",
        "//core/framework:framework",
        "//core/util:util",
    ],
    visibility = ["//visibility:public"],
)

