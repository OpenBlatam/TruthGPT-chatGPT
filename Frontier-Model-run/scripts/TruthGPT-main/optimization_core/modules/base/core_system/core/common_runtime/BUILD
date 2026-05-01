load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "common_runtime",
    srcs = [
        "base.py",
        "config.py", 
        "utils.py",
        "validation.py",
    ],
    deps = [
        "//core/platform:platform",
        "//core/util:util",
    ],
    visibility = ["//visibility:public"],
)

