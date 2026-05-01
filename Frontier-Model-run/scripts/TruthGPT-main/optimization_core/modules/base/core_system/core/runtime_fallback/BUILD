load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "runtime_fallback",
    srcs = [
        "realtime_optimizer.py",
    ],
    deps = [
        "//core/common_runtime:common_runtime",
        "//core/framework:framework",
        "//core/util:util",
    ],
    visibility = ["//visibility:public"],
)

